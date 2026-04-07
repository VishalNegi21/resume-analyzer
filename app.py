
import streamlit as st
import pdfplumber
import re
import nltk


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util

# ===== SETUP =====
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ===== FUNCTIONS =====
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Corrected regex and escaped backslash
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def tfidf_similarity(resume, job):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume, job])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return float(score[0][0]) * 100

model = SentenceTransformer('all-MiniLM-L6-v2')

def bert_similarity(resume, job):
    emb1 = model.encode(resume, convert_to_tensor=True)
    emb2 = model.encode(job, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0][0]) * 100

def skill_match_score(resume_skills, job_skills):
    if len(job_skills) == 0:
        return 0
    return (len(set(resume_skills) & set(job_skills)) / len(job_skills)) * 100

def final_score(tfidf_score, bert_score, skill_score):
    return 0.3 * tfidf_score + 0.5 * bert_score + 0.2 * skill_score

skills_dict = {
    "programming": ["python", "java", "c++", "javascript"],
    "data": ["sql", "excel", "tableau", "power bi"],
    "aiml": ["machine learning", "deep learning", "nlp"],
    "web": ["html", "css", "react", "node"],
    "cloud": ["aws", "docker", "kubernetes"]
}

def get_all_skills(skills_dict):
    skills = []
    for cat in skills_dict.values():
        skills.extend(cat)
    return skills

skills_list = get_all_skills(skills_dict)

def extract_skills_from_text(text, skills_list):
    text = text.lower()
    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return list(set(found))

def categorize_missing_skills(missing, skills_dict):
    result = {}
    for cat, skills in skills_dict.items():
        gap = [s for s in missing if s in skills]
        if gap:
            result[cat] = gap
    return result

# ===== UI =====
st.title("AI Resume Analyzer")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if resume_file and job_desc:

        resume_text = extract_text_from_pdf(resume_file)

        resume_clean = preprocess(resume_text)
        job_clean = preprocess(job_desc)

        tfidf_score = tfidf_similarity(resume_clean, job_clean)
        bert_score = bert_similarity(resume_text, job_desc)

        resume_skills = extract_skills_from_text(resume_text, skills_list)
        job_skills = extract_skills_from_text(job_desc, skills_list)

        missing = list(set(job_skills) - set(resume_skills))
        category_gap = categorize_missing_skills(missing, skills_dict)

        skill_score = skill_match_score(resume_skills, job_skills)

        score = final_score(tfidf_score, bert_score, skill_score)

        st.subheader("Match Score")
        st.progress(int(score))
        st.write(f"{score:.2f}% Match")

        st.write("Your Skills:", resume_skills)
        st.write("Missing Skills:", missing)
        st.write("Skill Gap:", category_gap)

        st.write("TF-IDF:", tfidf_score)
        st.write("BERT:", bert_score)
        st.write("Skill Match:", skill_score)

    else:
        st.error("Upload resume and job description")
