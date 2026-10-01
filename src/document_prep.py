import pandas as pd
import re
import os
import glob
from pathlib import Path
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_CHUNK_SIZE = 256
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def process_clinical_transcription(transcript_text):
    header_pattern = re.compile(r'(?m)^([A-Z\s/]+:)\s*')
    parts = header_pattern.split(str(transcript_text))
    
    final_chunks = []
    current_header = "UNSPECIFIED:"
    
    for part in parts:
        if header_pattern.match(part):
            current_header = part.strip()
        elif part.strip(): 
            content = part.strip()
            full_text = f"{current_header} {content}"
            
            if len(tokenizer.encode(full_text, add_special_tokens=False, truncation=True, max_length=10000)) > MAX_CHUNK_SIZE:
                # Calculate how many tokens the header consumes
                header_tokens = len(tokenizer.encode(f"{current_header} ", add_special_tokens=False))
                
                # Deduct header size from max limit so the final combined chunk fits safely
                available_chunk_size = max(50, MAX_CHUNK_SIZE - header_tokens)
                
                # Reintroduce Langchain splitter for recursive, boundary-aware splitting with overlap
                section_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=available_chunk_size,
                    chunk_overlap=30,
                    separators=["\n\n", "\n", ". ", ", ", " "],
                    length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False, truncation=True, max_length=10000))
                )
                
                # Split the raw content, then attach the header to every resulting piece
                content_chunks = section_splitter.split_text(content)
                for chunk in content_chunks:
                    final_chunks.append(f"{current_header} {chunk}")
            else:
                final_chunks.append(full_text)
            
    return final_chunks

def process_single_record(row: dict) -> dict:
    record_buckets = {}

    def get_text(key: str) -> str:
        val = row.get(key)
        return str(val).strip() if pd.notna(val) and val is not None else ""

    specialty = get_text('medical_specialty')
    if specialty:
        tokens = tokenizer.encode(specialty, add_special_tokens=False, truncation=True, max_length=10000)
        record_buckets['medical_specialty'] = [tokenizer.decode(tokens[:MAX_CHUNK_SIZE])]
    else:
        record_buckets['medical_specialty'] = []

    transcript = get_text('transcription')
    record_buckets['transcription'] = process_clinical_transcription(transcript) if transcript else []

    return record_buckets

def prepare_all_chunks(df):
    flattened_chunks = []
    
    for _, row in df.iterrows():
        buckets = process_single_record(row)
        
        for bucket_name, chunks in buckets.items():
            for chunk_pos, chunk_text in enumerate(chunks):
                flattened_chunks.append({
                    "encounter_id": row['encounter_id'],
                    "patient_id": row['patient_id'],
                    "visit_date": row['visit_date'],
                    "visit_number_for_patient": row['visit_number_for_patient'],
                    "bucket_name": bucket_name,
                    "chunk_position": chunk_pos,
                    "text": chunk_text
                })
                
    return flattened_chunks

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "unstructured" / "raw"
    processed_dir = base_dir / "data" / "unstructured" / "processed"
    output_path = processed_dir / "chunk_metadata.csv"
    
    visit_files = glob.glob(str(raw_dir / "visit_*_notes.csv"))
    
    if not visit_files:
        print("No visit files found.")
        exit()

    dfs = [pd.read_csv(f) for f in visit_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    gt_df = pd.read_csv(raw_dir / "ground_truth_labels.csv")
    master_df = combined_df.merge(gt_df[['encounter_id', 'visit_number_for_patient']], on='encounter_id', how='left')
    
    master_df['token_count'] = master_df['transcription'].astype(str).apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False, truncation=True, max_length=10000))
    )
    master_df = master_df[master_df['token_count'] >= 20]

    processed_dir.mkdir(parents=True, exist_ok=True)
    append_mode = False
    
    if output_path.exists():
        existing_metadata = pd.read_csv(output_path)
        if 'encounter_id' in existing_metadata.columns:
            existing_encounters = set(existing_metadata['encounter_id'].unique())
            master_df = master_df[~master_df['encounter_id'].isin(existing_encounters)]
            print(f"Skipping {len(existing_encounters)} existing encounters.")
            append_mode = True
        else:
            print("Legacy chunk_metadata.csv detected. Overwriting with new schema.")
            append_mode = False

    if master_df.empty:
        print("No new encounters to process.")
        exit()

    print(f"Executing chunking for {len(master_df)} new encounters...")
    new_chunks = prepare_all_chunks(master_df)
    new_chunks_df = pd.DataFrame(new_chunks)

    if append_mode:
        new_chunks_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        new_chunks_df.to_csv(output_path, index=False)
    
    print(f"Success! Formatted chunk metadata saved to {output_path}")