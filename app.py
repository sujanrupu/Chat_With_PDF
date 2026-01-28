import streamlit as st

# Below three are used for WebUI, such as buttons, text boxes, file upload
import os # Store environment variables
import tempfile # Store PDF temporarily
import time # Measure response time

from dotenv import load_dotenv # Loads.env file

from langchain_groq import ChatGroq # Connects to Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter # Breaks the PDF insmaller Chunks
from langchain_core.prompts import ChatPromptTemplate # Controls LLM query and response behavior
from langchain_classic.chains.combine_documents import create_stuff_documents_chain 
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS # Conducts similarity search
from langchain_community.document_loaders import PyPDFLoader # Load and reads text from PDF
from langchain_community.embeddings import HuggingFaceEmbeddings # Converts text to vector numbers

from nltk.tokenize import sent_tokenize # Sentence splitting
import nltk

def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

download_nltk_data()


# Load environment variables 
load_dotenv()

#  Streamlit UI (Sujan Ghosh)
st.set_page_config(page_title="Chat with PDF", layout="centered")
st.title("📄 Chat with PDF (Langchain + HuggingFace + Groq)")

#  LLM 
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

#  Prompt 
prompt = ChatPromptTemplate.from_template("""
Answer the question strictly using the provided context.
If the answer is not in the context, say "I don't know".

<context>
{context}
</context>

Question: {input}
""")

#  File Upload (Sujan Ghosh)
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

#  Vector Store Creation 
def create_vector_store(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path) # Converts PDF -> Text Pages
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter( # Divides the PDF into smaller chunks
        chunk_size=300,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # Vector embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings) # PDF summary for better similarity search
    return vectorstore

# Create Embeddings Button 
if uploaded_file and st.button("Create Embeddings"):
    with st.spinner("Processing PDF..."):
        st.session_state.vectors = create_vector_store(uploaded_file) # Because of using seassion state the PDF will be processed one using streamlit
    st.success("✅ PDF processed successfully!")

# Question Input
question = st.text_input("Ask a question from the PDF")

# Answer Generation (Sujan Ghosh)
if question and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 1})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": question})

    st.subheader("🧠 Answer")
    answer = response["answer"].strip()
    st.write(answer)

    st.caption(f"⏱ Response time: {time.process_time() - start:.2f}s")

    normalized_answer = answer.lower()

    # Show context ONLY if model did NOT say "I don't know"
    if "i don't know" not in normalized_answer:
        top_chunk = response["context"][0] if response["context"] else None

        with st.expander("📌 Retrieved Context"):
            if top_chunk:
                sentences = sent_tokenize(top_chunk.page_content)
                relevant_sentences = [
                    s for s in sentences
                    if any(w in s.lower() for w in question.lower().split())
                ]

                if relevant_sentences:
                    for s in relevant_sentences:
                        st.write(s)
                        st.write("---")
                else:
                    st.write(top_chunk.page_content)
            else:
                st.write("No relevant context found.")


elif question:
    st.warning("Please upload a PDF and create embeddings first.")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #6c757d;
        text-align: center;
        font-size: 14px;
        padding: 10px 0;
    }
    </style>

    <div class="footer">
        Developed by <strong>Sujan Ghosh</strong>
    </div>
    """,
    unsafe_allow_html=True
)

