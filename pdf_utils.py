# This file contains utilities for PDF processing, text extraction, and vector store creation
# Handles PDF reading, text chunking, and embeddings generation

import os
from typing import List, Optional
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found. AI features might not work.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Error configuring Google Generative AI: {e}")


def get_pdf_text(pdf_files_paths: List[str]) -> str:
    text = ""
    for pdf_file_path in pdf_files_paths:
        try:
            with open(pdf_file_path, "rb") as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n" # Add newline between pages
        except Exception as e:
            print(f"Error reading PDF {pdf_file_path}: {e}")
            continue
    return text.strip()

def get_text_chunks(text: str, chunk_size: int = 10000, chunk_overlap: int = 1000) -> List[str]:
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks: List[str], save_path: str) -> bool:
    if not text_chunks:
        print("No text chunks provided to create vector store.")
        return False
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(save_path)
        print(f"Vector store created successfully at {save_path}")
        return True
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return False


def load_vector_store(path: str) -> Optional[FAISS]:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # allow_dangerous_deserialization is necessary for FAISS with custom embeddings
        vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store loaded from {path}")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store from {path}: {e}")
        return None


def process_pdf_to_vector_store(pdf_paths: List[str], save_path: str) -> int:
    # Extract text from PDFs
    text = get_pdf_text(pdf_paths)
    
    if not text.strip():
        raise ValueError("No text could be extracted from the provided PDF(s).")
    
    # Create text chunks
    text_chunks = get_text_chunks(text)
    if not text_chunks:
        raise ValueError("Text was extracted, but no chunks were generated.")
        
    # Create and save vector store
    if not create_vector_store(text_chunks, save_path):
        raise ValueError("Failed to create and save the vector store.")
    
    return len(text_chunks)