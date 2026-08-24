import pandas as pd
import re
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_CHUNK_SIZE = 256

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

general_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_SIZE,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", ", ", " "],
    length_function=lambda x: len(tokenizer.encode(x, add_special_tokens=False))
)

def process_clinical_transcription(transcript_text):
    header_pattern = re.compile(r'([A-Z\s/]+:)\s*')
    parts = header_pattern.split(str(transcript_text))
    
    sections = []
    current_header = "UNSPECIFIED:"
    for part in parts:
        if header_pattern.match(part):
            current_header = part.strip()
        elif part.strip(): 
            sections.append(f"{current_header} {part.strip()}")
            
    final_chunks = []
    for section in sections:
        if len(tokenizer.encode(section, add_special_tokens=False)) > MAX_CHUNK_SIZE:
            final_chunks.extend(general_splitter.split_text(section))
        else:
            final_chunks.append(section)
            
    return final_chunks


def process_single_record(row):
    record_buckets = {}
    
    desc_text = str(row.get('description', ''))
    
    index_val = row.get('index')
    if index_val is not None and pd.notna(index_val):
        desc_text = f"Index {index_val}: {desc_text}"
        
    record_buckets['description'] = general_splitter.split_text(desc_text) if desc_text else []
    
    specialty_text = str(row.get('medical_specialty', ''))
    if specialty_text:
        encoded_tokens = tokenizer.encode(specialty_text, add_special_tokens=False)
        safe_text = tokenizer.decode(encoded_tokens[:MAX_CHUNK_SIZE])
        record_buckets['medical_specialty'] = [safe_text]
    else:
        record_buckets['medical_specialty'] = []
    
    transcript_text = str(row.get('transcription', ''))
    record_buckets['transcription'] = process_clinical_transcription(transcript_text) if transcript_text else []
    
    sample = str(row.get('sample_name', ''))
    keywords = str(row.get('keywords', ''))
    combined_text = f"Sample Name: {sample} | Keywords: {keywords}"
    record_buckets['sample_name_keywords'] = general_splitter.split_text(combined_text) if sample or keywords else []
    
    return record_buckets


def prepare_all_chunks(df):
    flattened_chunks = []
    
    for idx, row in df.iterrows():
        buckets = process_single_record(row)
        
        for bucket_name, chunks in buckets.items():
            for chunk_pos, chunk_text in enumerate(chunks):
                flattened_chunks.append({
                    "record_index": idx,
                    "bucket_name": bucket_name,
                    "chunk_position": chunk_pos,
                    "text": chunk_text
                })
                
    return flattened_chunks