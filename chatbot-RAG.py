

# STEP 1: Install Required Libraries

!pip install -q chromadb langchain pypdf gradio langchain-community #using langchain to integrate RAG based chat bot
!pip install -q google-generativeai langchain-google-genai #Gemini LLM
!pip install -q sentence-transformers  # Hugging Face embeddings


# STEP 2: Import Libraries

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import gradio as gr


# STEP 3: Setup Google Gemini API Key

from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")


# STEP 4: Load and Split PDF

pdf_path = "/hr_policy.pdf"  # Upload your PDF here /hr_policy.pdf
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)


# STEP 5: Create Embeddings + Vector Store (Hugging Face)

# Using a free Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    docs, embeddings, collection_name="hr_policy_hf_embeddings"
)


# STEP 6: Create QA Chain (Gemini LLM) #connect to LLM

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.5)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)


# STEP 7: Gradio Chatbot (Bigger Textboxes)

def chatbot(query):
    try:
        return qa_chain.run(query)
    except Exception as e:
        return f"Error: {e}"

demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="Ask HR Assistant a question", lines=3, placeholder="Type your HR question here..."),
    outputs=gr.Textbox(label="Answer", lines=12),
    title="AI-Powered HR Assistant"
)

demo.launch(share=True) #public link for chat bot will be genertaed
