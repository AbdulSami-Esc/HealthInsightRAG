import pandas as pd
import os
import faiss
from collections import defaultdict

from rank_bm25 import BM25Okapi
import numpy as np


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

##Definig the function load_snippets()
##Loads the procesed dataset and return the dataframe
def load_snippets(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'processed', filename)

    data=pd.read_csv(file_path)

    print("Shape:",data.shape)
    print("Columns:",data.columns.tolist())
    return data

##convert the all the strings in simple embeddings and then a list 
def generate_embeddings(df):
    snippet_list=df["Snippet"].fillna("").astype(str).tolist()

    ##Pass whole series os string to encode at once so that CPU can do hardware acceleration
    vec=model.encode(snippet_list,batch_size=128,show_progress_bar=True)
    print(type(vec), vec.shape, vec.dtype)

    return vec

##build IndexFlatL2 with dimensions inferred from embeddings shape
def build_faiss_index(embeddings):
    dimensions=embeddings.shape[1]

    index=faiss.IndexFlatL2(dimensions)

    index.add(embeddings)

    return index


##It would conver user query in integer index and then take it to our existing embeddings array
##Then resues that embeddings ss query
def search_index(index, embeddings, df, query_position=0, k=5):
    query = embeddings[query_position].reshape(1, -1)
    distances, indices = index.search(query, k)

    print("Indices:", indices)
    print("Distances:", distances)

    for i in indices[0]:
        print(df['Snippet'].iloc[i])

    return distances, indices

def save_index(index, filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, '..', 'data', 'processed', filename)

    faiss.write_index(index, output_file)
    print(f"Index saved to {output_file}")


def load_index(path):
    return faiss.read_index(path)

def build_bm25_index(df):
    snippets = df["Snippet"].fillna("").astype(str).str.lower().tolist()
    tokenized_corpus = [doc.split() for doc in snippets]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def embed_query(query_text):
    # model.encode([query_text]) -> single vector, reshaped for FAISS
    # model.encode expects a list of strings and returns a 2D numpy array
    vector = model.encode([query_text], convert_to_numpy=True)
    
    return vector.astype('float32')


def reciprocal_rank_fusion(faiss_ranked_indices, bm25_ranked_indices, k=60):
    fused_scores = defaultdict(float)
    
    # Process FAISS rankings (1-based rank)
    for rank, idx in enumerate(faiss_ranked_indices, start=1):
        fused_scores[idx] += 1.0 / (k + rank)
        
    # Process BM25 rankings
    for rank, idx in enumerate(bm25_ranked_indices, start=1):
        fused_scores[idx] += 1.0 / (k + rank)
        
    # Sort indices by their fused score in descending order
    sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    
    return sorted_indices



def hybrid_search(query_text, faiss_index, bm25_index, df, k=5, candidates=20):
    # 1. Dense Search(FAISS changes it into dense vector and feches top 20 semntical closest row)
    query_vector = embed_query(query_text) # Shape: (1, 384)
    distances, faiss_indices = faiss_index.search(query_vector, candidates)
    faiss_ranked_indices = faiss_indices[0].tolist() # Flatten 2D array to 1D list
    
    # 2. Sparse Search (BM25)
    # Must use the exact same tokenization style as build_bm25_index (.lower().split())
    tokenized_query = query_text.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    # Get indices of top candidates sorted by score descending
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:candidates].tolist()
    
    #Combine Rankings using Reciprocal Rank Fusion
    fused_indices = reciprocal_rank_fusion(faiss_ranked_indices, bm25_ranked_indices)
    
    #Take the top 'k' final results
    top_k_indices = fused_indices[:k]
    
    #Return the matching rows from your dataset for your FastAPI route to process
    return df.iloc[top_k_indices]    






if __name__ == "__main__":
    # Load processed snippets
    dataframe = load_snippets('snippets.csv')
    print(dataframe.head())

    # Dense index: embeddings + FAISS
    embeddings = generate_embeddings(dataframe)
    faiss_idx = build_faiss_index(embeddings)
    print(faiss_idx)
    save_index(faiss_idx, 'faiss_index.bin')

    # Sparse index: BM25 (rebuilt from snippets each run, not saved to disk)
    bm25_idx = build_bm25_index(dataframe)

    # Sanity check: run a real hybrid query before wiring this into the orchestrator
    test_query = "diabetic patient emergency admission"
    results = hybrid_search(test_query, faiss_idx, bm25_idx, dataframe, k=5)
    print(results[["Snippet"]])


    
