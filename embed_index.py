from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import pandas as pd

def generate_embeddings(texts, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_index(index, path):
    faiss.write_index(index, path)

def load_index(path):
    return faiss.read_index(path)

if __name__ == "__main__":
    df = pd.read_csv("kcc_preprocessed.csv")
    df = df.dropna(subset=['question', 'answer'])

    texts = (df['question'].astype(str) + " " + df['answer'].astype(str)).tolist()

    embeddings = generate_embeddings(texts)
    index = build_faiss_index(embeddings)
    save_index(index, "kcc_faiss.index")
    with open("kcc_texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    print("Embeddings and FAISS index saved.")