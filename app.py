import streamlit as st
import tempfile
import pdfplumber
import docx2txt
import spacy
import os
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text
import json

# Load spaCy model
import spacy
import subprocess
import sys
import streamlit as st

@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()



# Load tips.json from uploaded file path
@st.cache_resource
def load_tips():
    with open("tips.json", "r") as f:
        return json.load(f)

keyword_tips = load_tips()

def extract_text(file):
    text = ""

    if file.name.lower().endswith(".pdf"):
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            st.error(f"PDF parsing error: {e}")

    elif file.name.lower().endswith(".docx"):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                file.seek(0)
                tmp.write(file.read())
                tmp_path = tmp.name

            text = docx2txt.process(tmp_path)

        except Exception as e:
            st.error(f"DOCX parsing error: {e}")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return text


def score_resume(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text.lower(), jd_text.lower()])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return round(float(similarity[0][0]) * 100, 2)

def suggest_improvements(resume_text, jd_text):
    resume_doc = nlp(resume_text)
    jd_doc = nlp(jd_text)

    resume_tokens = {
        token.lemma_.lower()
        for token in resume_doc 
        if token.is_alpha
    }
    jd_tokens = {
        token.lemma_.lower()
        for token in jd_doc 
        if token.is_alpha
    }

    missing_keywords = jd_tokens - resume_tokens
    suggestions = sorted([word for word in missing_keywords if len(word) > 3])[:10]
    return suggestions


def categorize_keywords(keywords):
    skills = {"python", "sql", "excel", "tableau", "powerbi", "ml", "data", "statistics"}
    tools = {"jira", "github", "snowflake", "docker", "tensorflow"}
    project_keywords = {"analysis", "prediction", "dashboard", "model", "report", "pipeline"}

    categorized = {"skills": [], "tools": [], "projects": [], "other": []}

    for word in keywords:
        if word in skills:
            categorized["skills"].append(word)
        elif word in tools:
            categorized["tools"].append(word)
        elif word in project_keywords:
            categorized["projects"].append(word)
        else:
            categorized["other"].append(word)

    return categorized

# ------------------------ UI ------------------------

st.title("🔍 Resume & JD Matcher")
st.markdown("""
Upload your **resume** and **job description** below to get a match score and personalized improvement suggestions.
""")

st.header("📄 Upload Resume")
st.caption("Accepted formats: PDF, DOCX | Max size: 200MB")
resume_file = st.file_uploader("", type=["pdf", "docx"])

st.header("📝 Paste Job Description")
jd_input = st.text_area("Paste the job description here")

if resume_file and jd_input.strip():
    st.success("✅ Resume and Job Description uploaded!")
    
if st.button("🔍 Analyze Resume"):
    resume_text = extract_text(resume_file)

    if not resume_text.strip():
        st.error("❌ Could not extract text from resume. Please try another file.")
    else:
        score = score_resume(resume_text, jd_input)
        suggestions = suggest_improvements(resume_text, jd_input)
        categories = categorize_keywords(suggestions)

        # 🎯 Resume Match Score Section
        st.subheader("📊 Resume Match Score")

        if score > 80:
            label = "🌟 Excellent Match"
        elif score > 60:
            label = "✅ Good Match"
        elif score > 40:
            label = "⚠️ Fair Match"
        else:
            label = "❌ Weak Match"

        st.metric("Match %", f"{score:.1f}%", label)
        st.progress(score / 100)

        # 🛠 Suggestions Section
        st.subheader("🛠 Suggestions to Improve")

        has_suggestions = any(categories.values())
        if has_suggestions:
            if categories["skills"]:
                st.info("**Skills to Add:** " + ", ".join(categories["skills"]))
            if categories["tools"]:
                st.info("**Tools/Technologies to Mention:** " + ", ".join(categories["tools"]))
            if categories["projects"]:
                st.info("**Suggested Project Topics:** " + ", ".join(categories["projects"]))
            if categories["other"]:
                st.info("**Other Relevant Terms:** " + ", ".join(categories["other"]))
        else:
            st.success("🎉 Your resume already covers most relevant keywords!")

        # 💡 Improvement Tips Section
        st.subheader("💡 Improvement Guidance")
        shown_tips = set()

        for word in categories["skills"] + categories["tools"] + categories["projects"] + categories["other"]:
            tip = keyword_tips.get(word)
            if tip and tip not in shown_tips:
                st.markdown(f"- {tip}")
                shown_tips.add(tip)

        if not shown_tips:
            st.success("✅ You're well aligned with the job description!")

        # 📝 Logging Section
        log_entry = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "match_score": score,
            "missing_keywords": ", ".join(suggestions)
        }])
        log_file = "evaluation_log.csv"
        log_entry.to_csv(log_file, mode="a", header=not os.path.exists(log_file), index=False)
