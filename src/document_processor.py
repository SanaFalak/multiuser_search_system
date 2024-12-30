# src/document_processor.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def process_pdf(self, file_path: str) -> list:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return self.text_splitter.split_text(text)
        except FileNotFoundError:
            raise FileNotFoundError(f"Document not found at {file_path}")