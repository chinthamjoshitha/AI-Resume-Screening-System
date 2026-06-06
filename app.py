import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.set_page_config(page_title="AI Resume Screening System")

st.title("AI Resume Screening System")

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

uploaded_resume = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Enter Job Description"
)

if uploaded_resume and job_description:

    resume_text = extract_text_from_pdf(uploaded_resume)

    documents = [resume_text, job_description]

    cv = CountVectorizer()

    matrix = cv.fit_transform(documents)

    similarity = cosine_similarity(matrix)[0][1]

    score = round(similarity * 100, 2)

    st.subheader("Resume Match Score")
    st.success(f"{score}%")

    skills = [
        "python",
        "sql",
        "machine learning",
        "data analysis",
        "pandas",
        "numpy",
        "streamlit",
        "tensorflow",
        "power bi",
        "excel",
        "git"
    ]

    missing_skills = []

    resume_lower = resume_text.lower()

    for skill in skills:
        if skill not in resume_lower:
            missing_skills.append(skill)

    st.subheader("Missing Skills")

    if missing_skills:
        for skill in missing_skills:
            st.write("❌", skill)
    else:
        st.success("No Missing Skills Found")