
import os
from dotenv import load_dotenv

# Loaders & splitters 
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store
from langchain_community.vectorstores import Chroma  

# Self-query & schema tools 
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# ====== GEMINI: Chat + Embeddings ======
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ----------------- Keys -----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # ensure env var is set

# ----------------- Models ----------------
# Embeddings: Google Text Embedding 004
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
                                         credentials={"api_key":GOOGLE_API_KEY})

# Chat LLM: Gemini 1.5 Flash (fast & capable). You can switch to "gemini-1.5-pro".
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

# ----------------- Functions -------------
# Load resumes in different formats (unchanged)
def load_resume(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        import docx2txt  # ensure installed
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format.")
    return loader.load()

# Analyze the resume using Gemini (minimal change: llm now Gemini)
def analyze_resume(docs, job_description):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    full_analysis = ""
    for chunk in chunks:
        prompt = f"""
Compare this resume with the job description. Give:
1. Suitability Score (out of 100)
2. Skills Matched
3. Experience Relevance
4. Education Evaluation
5. Strengths
6. Weaknesses
7. Final Recommendation

Job Description:
{job_description}

Resume:
{chunk.page_content}
"""
        result = llm.invoke(prompt)  # Gemini call
        full_analysis += result.content + "\n\n"
    return full_analysis

# Store text chunks into ChromaDB (embeddings now Google)
def store_to_vectorstore(text_chunks, persist_directory="chroma_store"):
    texts = [chunk.page_content for chunk in text_chunks]
    metadatas = [{"source": f"resume_chunk_{i}"} for i in range(len(texts))]

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embedding,                 # Google embeddings
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

# Use SelfQueryRetriever to interpret and fetch relevant chunks (llm now Gemini)
def run_self_query(query, persist_directory="chroma_store"):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding         # Google embeddings
    )

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="Where the chunk is from",
            type="string"
        )
    ]

    document_content_description = "This represents a chunk of a resume."

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        search_type="mmr"
    )

    return retriever.get_relevant_documents(query)

# ===================== END OF resume_processor =====================



import streamlit as st
from resume_processor import load_resume, analyze_resume, store_to_vectorstore, run_self_query
import os

st.set_page_config(page_title="AI Resume Screener")
#st.write("streamlit is rendering")
st.title("AI Resume Screener")
st.markdown("Upload a resume and analyze it using AI. Then run smart searches over previous resumes.")

job_desc = st.text_area("Paste Job Description")
uploaded_file = st.file_uploader("📎 Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if st.button("Analyze & Store") and uploaded_file and job_desc:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing & Storing Resume..."):
        docs = load_resume(uploaded_file.name)
        report = analyze_resume(docs, job_desc)  # uses Gemini inside resume_processor.py
        store_to_vectorstore(docs)
        st.success("✅ Analysis complete and stored!")

        st.subheader("📄 AI Resume Summary")
        st.write(report)
        st.download_button("📥 Download Report", report, file_name="resume_analysis.txt")

st.divider()
st.subheader("🔎 Ask Anything About Stored Resumes")
query = st.text_input("Type your smart query here (e.g., 'Python developer with AWS')")

if st.button("Search Resumes") and query:
    with st.spinner("Searching..."):
        results = run_self_query(query)
        if results:
            for i, res in enumerate(results, 1):
                st.markdown(f"**Result {i}:**")
                st.write(res.page_content.strip())
        else:
            st.warning("No matches found.")
