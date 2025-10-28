from pypdf import PdfReader
import os

def load_pdf_documents(folder_path: str) -> str:
    text_corpus = ""
    for path in os.listdir(folder_path):
        path = os.path.join(folder_path, path)
        if path.endswith(".pdf"):
            text = load_pdf(path)
            text_corpus += text
    return text_corpus

def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
