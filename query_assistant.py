import pickle
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

class KCCQueryAssistant:
    def __init__(self, index_path, texts_path, model_api_url=None):
        self.index = faiss.read_index(index_path)
        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)
        self.model_api_url = model_api_url
        self.embed_model = SentenceTransformer("all-mpnet-base-v2")

    def embed_query(self, query):
        return self.embed_model.encode([query], convert_to_numpy=True)[0]

    def semantic_search(self, query, top_k=5, threshold=0.6):
        query_vec = self.embed_query(query)
        distances, indices = self.index.search(np.array([query_vec]), top_k)
        print("Distances:", distances)
        print("Indices:", indices)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if dist < threshold:
                print(f"[Match] {self.texts[idx][:100]}...")  # Preview matched text
                results.append((self.texts[idx], dist))
        return results

    def call_local_llm(self, prompt):
        # This is a mock LLM. Just returns the first sentence from context
        if self.model_api_url:
            response = requests.post(self.model_api_url, json={"prompt": prompt})
            return response.json().get("text", "LLM Error")
        else:
            # Simulate answer: Return first line of context
            context = prompt.split("Context:")[1].split("Question:")[0].strip()
            return context.split("\n")[0] if context else "No context found."

    def query(self, user_query):
        if not user_query.strip():
            return {
                "source": "Error",
                "answer": "Please enter a valid question.",
                "question": user_query
            }

        results = self.semantic_search(user_query)

        if results:
            context = "\n".join([r[0] for r in results])
            prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
            answer = self.call_local_llm(prompt)
            return {
                "source": "KCC dataset",
                "answer": answer.strip(),
                "question": user_query
            }
        else:
            fallback_answer = self.internet_search_fallback(user_query)
            return {
                "source": "Internet fallback",
                "answer": fallback_answer,
                "question": user_query
            }

    def internet_search_fallback(self, query):
        return f"No local context found for '{query}'. Performing live Internet search is not implemented here."

if __name__ == "__main__":
    assistant = KCCQueryAssistant("kcc_faiss.index", "kcc_texts.pkl", model_api_url=None)
    while True:
        q = input("Ask KCC Assistant > ")
        if q.lower() in ["exit", "quit"]:
            break
        result = assistant.query(q)
        print(f"[{result['source']}] {result['answer']}")
