import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp{
    background-color:#FAF7FF;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#6A4C93;
}

[data-testid="stSidebar"] *{
    background-color:#6A4C93;
    width:260px;
}


/* Buttons */
.stButton>button{
    background-color:#B185DB;
    color:white;
    border-radius:12px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background-color:#9D6BCF;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("📌IMDb Analyzer")

page = st.sidebar.radio(
    "",
    ["Home", "🎥 Predict Review", " About"]
)

if page == "Home":

    st.title("🎬 IMDb Movie Reviews Sentiment Analysis")

    st.markdown("### Powered by NLP & Machine Learning")

    st.info("""
!! Welcome !!

This application here predicts whether an IMDb movie review is **Positive** or **Negative** using Natural Language Processing(NLP) and Machine Learning.
""")

    st.markdown("""
### Features

- Predict Movie Review Sentiment
- Interactive Streamlit Interface
""")

elif page == "🎥 Predict Review":

    st.title("🎥 Predict Movie Review")

    review = st.text_area(
        "Enter your review",
        placeholder="Example: This movie was amazing! / It was a bad movie."
    )

    if st.button(" Analyze Review"):

        if review.strip() == "":
            st.warning("Please enter a review.")

        else:

            review_vector = vectorizer.transform([review])

            prediction = model.predict(review_vector)[0]

            if prediction == "positive":

                st.success("Positive Review")

            else:

                st.error("!!Negative Review!!")

elif page == " About":

    st.title(" About this Project")

    st.markdown("""
### 👩‍💻 IMDb Movie Review Sentiment Analysis

** 🛠 Machine Learning Model:** Logistic Regression

**Vectorization:** TF-IDF

**Accuracy:** **90.56%**

**Dataset:** IMDb Movie Reviews Dataset

**Developer:** Riya Prashant Bhosle
""")

st.markdown("---")
st.caption("Built using Python • Streamlit • TF-IDF • Logistic Regression")