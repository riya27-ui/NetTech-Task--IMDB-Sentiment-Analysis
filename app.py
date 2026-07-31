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

.stApp{
    background-color:#FAF7FF;
}
[data-testid="stAppViewContainer"]{
    color:#222222;
}

[data-testid="stSidebar"]{
    background-color:#6A4C93;
}

[data-testid="stSidebar"] *{
    color:white;
}

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
st.markdown("""
<style>

.stApp{
    background-color:#FAF7FF;
}

[data-testid="stSidebar"]{
    background-color:#6A4C93;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color:white !important;
}

.block-container{
    padding-top:3rem;
}

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
    ["Home", "🎥 Predict Review","Upload Dataset", "About"]
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

    st.title("🎥 Predict a Movie Review")

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

elif page == "Upload Dataset":

    st.title("📂 Upload Dataset")

    st.markdown("---")
    st.subheader("📂 Upload a Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing movie reviews",
        type=["csv"]
    )

    if uploaded_file is not None:
        import pandas as pd

        df = pd.read_csv(uploaded_file)

        st.write("Dataset Preview")
        st.dataframe(df.head())

        if "review" not in df.columns:
            st.error("CSV must contain a column named 'review'")
        else:
            if st.button("Predict Dataset"):

                reviews = vectorizer.transform(df["review"])

                predictions = model.predict(reviews)

                df["Prediction"] = predictions

                st.success("Prediction Completed!")

                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇ Download Results",
                    csv,
                    "predicted_reviews.csv",
                    "text/csv"
                )

elif page == "About":

    st.title("About this Project")

    st.markdown("""
### 👩‍💻 IMDb Movie Review Sentiment Analysis 👩‍💻 

 **🛠 Machine Learning Model:**  Logistic Regression

**Vectorization:** TF-IDF

**Accuracy:** **91.01%**

**Dataset:** IMDb Movie Reviews Dataset


**Developer:** Riya Prashant Bhosle
""")

st.markdown("---")
st.caption("Built using Python • Streamlit • TF-IDF • Logistic Regression")
