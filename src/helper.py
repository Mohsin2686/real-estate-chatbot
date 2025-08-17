import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings

# Load CSV file into Documents
def load_csv_file(file_path: str) -> List[Document]:
    """
    Load a CSV file and convert each row into a Document object.
    Metadata will contain the row index.
    """
    df = pd.read_csv(file_path)
    
    # Drop empty rows
    df = df.dropna()

    documents: List[Document] = []
    for idx, row in df.iterrows():
        # Convert entire row into a string
        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(
            Document(
                page_content=content,
                metadata={"row": idx}
            )
        )
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only 'row' in metadata and original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        row_idx = doc.metadata.get("row")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"row": row_idx}
            )
        )
    return minimal_docs


def text_split(extracted_data: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings