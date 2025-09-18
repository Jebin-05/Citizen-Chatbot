import os
import json
from langchain.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = FakeEmbeddings(size=384)

# Function to load JSON files and create documents
def load_json_files(file_paths):
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Create content based on keys
                    content = ""
                    metadata = {}
                    for key, value in item.items():
                        if key.lower() in ['content', 'description', 'details', 'response']:
                            content += str(value) + " "
                        else:
                            metadata[key] = str(value)
                    if content.strip():
                        doc = Document(page_content=content.strip(), metadata=metadata)
                        documents.append(doc)
                else:
                    # If item is not dict, treat as content
                    doc = Document(page_content=str(item), metadata={"source": file_path})
                    documents.append(doc)
        else:
            # If not list, treat as single document
            content = json.dumps(data, ensure_ascii=False)
            doc = Document(page_content=content, metadata={"source": file_path})
            documents.append(doc)
    
    return documents

# Function to split documents
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

# Main function to setup RAG
def setup_rag():
    # List of JSON files
    json_files = [
        'finetune_QA.json',
        'processed_rag_dept.json',
        'processed_rag_services.json',
        'rag_new_scheme.json',
        'tamil_scheme_data.json'
    ]
    
    # Load documents
    documents = load_json_files(json_files)
    print(f"Loaded {len(documents)} documents")
    
    # Split documents
    split_docs = split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Vector database created and persisted to ./chroma_db")
    return vectorstore

if __name__ == "__main__":
    setup_rag()
