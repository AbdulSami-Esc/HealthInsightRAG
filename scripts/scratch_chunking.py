import pandas as pd
import os
import re
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load 1-2 sample reports from the Kaggle dataset (plain text)
# Write a simple recursive_chunk(text, chunk_size, overlap) function yourself — split on paragraph → sentence → word, in that priority order (this is the "recursive" idea from the video)
# Print each chunk's length in tokens (use the same tokenizer logic you'd use for MiniLM) to confirm none exceed 256 tokens
# Manually inspect 3-4 chunks — check they still make sense standalone (no mid-sentence cuts, no lost context)
# Once this works standalone, we bring it into document_prep.py for real

def load_healthcare_samples(root_dir, filename="mtsamples.csv", num_samples=3):
    # Build the file path pointing to the data/raw/ directory shown in the screenshot
    file_path = os.path.join(root_dir, 'data','unstructured', 'raw', filename)
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
    
        return df.head(num_samples)
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return df


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def process_clinical_transcription(transcript_text, max_size):
    ##First pass to chunk y seection
    header_pattern = re.compile(r'([A-Z\s/]+:)\s*') ##Any section in column of description has topic nam ein capital letters and a colon
    parts = header_pattern.split(str(transcript_text))
    
    sections = []
    current_header = "UNSPECIFIED:"
    for part in parts:
        if header_pattern.match(part):
            current_header = part.strip()
        elif part.strip(): 
            sections.append(f"{current_header} {part.strip()}")
            
    # Added the lambda length_function to count tokens instead of characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size, 
        chunk_overlap=30, 
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False))
    )
    
    final_chunks = []
    for section in sections:
        # Replaced len(section) with the tokenizer length check so we evaluate token count
        if len(tokenizer.encode(section, add_special_tokens=False)) > max_size:
            final_chunks.extend(splitter.split_text(section))
        else:
            final_chunks.append(section)
    return final_chunks


def process_single_record(row, max_size=256):
    # General splitter for columns that just need standard length-based chopping
    # Added the lambda length_function here as well
    general_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False))
    )
    
    # This dictionary acts as our buckets
    record_buckets = {}
    
    # 1. Description (and Index if available)
    # The list inherently acts as the main chunk (index 0) and subchunk bucket (index 1+)
    desc_text = str(row.get('description', ''))
    if 'index' in row:
        desc_text = f"Index {row['index']}: {desc_text}"
    record_buckets['description'] = general_splitter.split_text(desc_text) if desc_text else []
    
    # 2. Medical Specialty
    # Rule: ONLY one chunk can be created. 
    specialty_text = str(row.get('medical_specialty', ''))
    if specialty_text:
        encoded_tokens = tokenizer.encode(specialty_text, add_special_tokens=False)
        safe_text = tokenizer.decode(encoded_tokens[:max_size])
        record_buckets['medical_specialty'] = [safe_text]
    else:
        record_buckets['medical_specialty'] = []
    
    # 3. Transcription
    # Rule: Use the 2-pass logic and store in a bucket exactly named 'transcription'
    transcript_text = str(row.get('transcription', ''))
    record_buckets['transcription'] = process_clinical_transcription(transcript_text, max_size) if transcript_text else []
    
    # 4. Keywords and Sample Name combined
    # Rule: Combine them and chunk them together
    sample = str(row.get('sample_name', ''))
    keywords = str(row.get('keywords', ''))
    combined_text = f"Sample Name: {sample} | Keywords: {keywords}"
    record_buckets['sample_name_keywords'] = general_splitter.split_text(combined_text) if sample or keywords else []
    
    return record_buckets


def main():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    samples = load_healthcare_samples(root_dir=".")

    for i, row in samples.iterrows():
        print(f"\n{'='*20} RECORD {i} {'='*20}")
        buckets = process_single_record(row, max_size=256)

        for bucket_name, chunks in buckets.items():
            print(f"\n--- Bucket: {bucket_name} ({len(chunks)} chunk(s)) ---")
            for j, chunk in enumerate(chunks):
                token_count = len(tokenizer.encode(chunk))
                status = "OK" if token_count <= 256 else "EXCEEDS LIMIT"
                print(f"  Chunk {j} | tokens={token_count} | {status}")
                print(f"  Text: {chunk[:150]}...")

if __name__ == "__main__":
    main()


