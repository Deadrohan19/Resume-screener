import streamlit as st
import joblib
import re
import pdfplumber  # For handling PDF files
from io import StringIO

def load_models():
    try:
        tfidf = joblib.load('tfidf.joblib')
        clf = joblib.load('clf.joblib')
        st.write("🎉 Success! Your AI is Ready to Go! 🎉")
        st.write("The TF-IDF vectorizer and classifier have been loaded successfully, and we're all set to help you unlock the potential of your resumes!")
        return tfidf, clf
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

tfidf, clf = load_models()


def clean_resume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', clean_text)  # Removed extra backslash
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def main():
    st.title("Resume Screening App")
    st.write("💼 Upload Your Resume Below:")
    st.write("Transform your skills into an opportunity with our powerful AI-powered Resume Screening App!")
    uploaded_file = st.file_uploader('', type=['txt', 'pdf'], label_visibility="collapsed")

    if uploaded_file is not None:
        # Read file content
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)

        if cleaned_resume.strip():  # Proceed only if there is text after cleaning
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]
            prediction_prob = clf.predict_proba(input_features)[0]

            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")
            confidence = prediction_prob[prediction_id] * 100  # Confidence level

            st.write("Predicted Category:", category_name)
            st.write(f"Confidence Level: {confidence:.2f}%")
        else:
            st.error("Uploaded resume appears to be empty. Please try again with a valid file.")

if __name__ == "__main__":
    main()
