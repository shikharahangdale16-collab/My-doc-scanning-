import os
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Extract Text from PDF ----------
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# ---------- Extract Text from DOCX ----------
def extract_text_from_docx(file_path):
    text = ""
    doc = docx.Document(file_path)
    for para in doc.paragraphs:
        text += para.text + " "
    return text

# ---------- Main Screening Function ----------
def screen_documents(job_description, folder_path):
    documents = []
    doc_names = []

    # Process each file in folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            continue

        documents.append(text)
        doc_names.append(file)

    # Insert Job Description at index 0
    documents.insert(0, job_description)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Cosine Similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Rank documents
    ranked_docs = sorted(list(zip(doc_names, similarity_scores)), key=lambda x: x[1], reverse=True)

    return ranked_docs


# ---------- Example Run ----------
if __name__ == "__main__":
    # Example Job Description
    job_desc = "Looking for a Python developer with skills in Django, SQL, and Machine Learning."

    # Folder containing resumes
    folder = "resumes"  # <-- Create a folder named 'resumes' and put PDFs/DOCX/TXT files

    results = screen_documents(job_desc, folder)

    print("\n📌 Document Screening Results:")
    for doc, score in results:
        print(f"{doc} --> Match Score: {score*100:.2f}%")
