import pandas as pd
import os
import faiss

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
    list=df["Snippet"].fillna("").astype(str).tolist()

    ##Pass whole series os string to encode at once so that CPU can do hardware acceleration
    vec=model.encode(list,batch_size=128,show_progress_bar=True)
    print(type(vec), vec.shape, vec.dtype)

    return vec

def build_faiss_index(embeddings):
    dimensions=embeddings.shape[1]

    index=faiss.IndexFlatL2(dimensions)

    index.add(embeddings)

    return index

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





if __name__=="__main__":
    ##Loading off the snippets
    dataframe=load_snippets('snippets.csv')
    print(dataframe.head()) ##checkign whther snippets are loaded

    ##Geenrate embeddings
    embeddings=generate_embeddings(dataframe)

    ##Store in FAISS index
    Index=build_faiss_index(embeddings)
    print(Index)

    ##Save th file 
    save_index(Index, 'faiss_index.bin')


    
