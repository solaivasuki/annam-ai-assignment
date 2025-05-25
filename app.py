import streamlit as st
from query_assistant import KCCQueryAssistant
import os

# Optional: suppress warnings
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "1"
os.environ["PYTORCH_NO_WATCH"] = "1"

# Cache assistant loading
@st.cache_resource
def load_assistant():
    return KCCQueryAssistant("kcc_faiss.index", "kcc_texts.pkl", model_api_url=None)

def main():
    st.title("🌾 KCC Query Assistant")
    assistant = load_assistant()

    query = st.text_input("Enter your agricultural question:")

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Searching KCC dataset..."):
                response = assistant.query(query)
            st.markdown(f"**🔎 Question:** {response.get('question', query)}")
            st.markdown(f"**📘 Answer:** {response['answer']}")
            st.markdown(f"**📚 Source:** {response['source']}")

if __name__ == "__main__":
    main()
