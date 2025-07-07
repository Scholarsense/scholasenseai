from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import fitz  # PyMuPDF to read PDFs
import pandas as pd

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")  # Extract text from each page
    return text

# Adjust PDF reading logic
pdf_folder = "x"  # Folder where PDFs are stored
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    # Loop over all PDF files in the folder
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            
            # Assuming each PDF contains Title, Review, Rating, and Date in a consistent format.
            # You may need to modify this based on your PDF structure.
            title = "Extracted Title"  # You can extract the title manually from the PDF content or metadata
            review = text  # Use all the text from the PDF as the review
            rating = "Rating info"  # Extract rating from PDF (may be in metadata or body)
            date = "Date info"  # Extract date info (could be in metadata or body)

            document = Document(
                page_content=title + " " + review,
                metadata={"rating": rating, "date": date},
                id=pdf_file  # Use the PDF filename as the unique ID
            )
            ids.append(pdf_file)
            documents.append(document)

# Initialize vector store (Chroma)
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents to the vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# Now you can use the retriever to query the database with the PDF documents.
