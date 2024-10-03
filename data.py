from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain.vectorstores import FAISS

# Load the PDF document (MIDA Malaysia)
loader = PyPDFLoader("MIDA.pdf")
docs = loader.load_and_split()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# Load Hugging Face model for embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate embeddings using Hugging Face model
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Create FAISS vector store (RAG component)
vectorstoredb = FAISS.from_documents(documents, embeddings)

# Save FAISS vector store to file for future use
vectorstoredb.save_local("faiss_store")













































# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OllamaEmbeddings

# # Load the PDF document (MIDA Malaysia)
# loader = PyPDFLoader("MIDA.pdf")
# docs = loader.load_and_split()

# # Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# documents = text_splitter.split_documents(docs)

# # Generate embeddings using Ollama
# embeddings = OllamaEmbeddings(model="llama3.1:8b")

# # Create FAISS vector store (RAG component)
# vectorstoredb = FAISS.from_documents(documents, embeddings)

# # Save FAISS vector store to file for future use
# vectorstoredb.save_local("faiss_store")