import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from preprocess import clean_text

st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    page_icon="chart_with_upwards_trend",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sentiment_model.pkl")

SENTIMENT_COLORS = {
    "Positive": "#1a9641",
    "Neutral": "#d4a017",
    "Negative": "#d73027",
}

SENTIMENT_ICONS = {
    "Positive": "Positive",
    "Neutral": "Neutral",
    "Negative": "Negative",
}

@st.cache_resource
def load_model(path: str):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

def predict(model, text: str) -> dict:
    if not text or len(text.strip()) == 0:
        raise ValueError("Review text cannot be empty.")

    cleaned = clean_text(text)
    if len(cleaned.strip()) == 0:
        raise ValueError("After cleaning, the review text is empty. Please provide a meaningful review.")

    prediction = model.predict([cleaned])[0]

    result = {
        "sentiment": prediction,
        "cleaned_text": cleaned,
        "confidence": None,
    }

    try:
        proba = model.predict_proba([cleaned])[0]
        classes = model.classes_
        result["confidence"] = dict(zip(classes, proba.round(4)))
    except AttributeError:
        pass

    return result

def render_sentiment_result(result: dict) -> None:
    sentiment = result["sentiment"]
    color = SENTIMENT_COLORS.get(sentiment, "#333333")

    st.markdown(
        f"""
        <div style="
            background-color: {color}18;
            border-left: 5px solid {color};
            border-radius: 6px;
            padding: 18px 22px;
            margin: 16px 0;
        ">
            <h3 style="color: {color}; margin: 0 0 6px 0;">
                Predicted Sentiment: {sentiment}
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["confidence"]:
        st.markdown("**Confidence Scores:**")
        conf = result["confidence"]
        ordered = ["Positive", "Neutral", "Negative"]

        for label in ordered:
            if label in conf:
                score = conf[label]
                col_color = SENTIMENT_COLORS.get(label, "#333")
                st.markdown(
                    f"""
                    <div style="margin-bottom: 6px;">
                        <span style="width: 80px; display: inline-block; font-weight: 600; color: {col_color}">
                            {label}
                        </span>
                        <span style="font-size: 13px; color: #666;">
                            {score * 100:.1f}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(score))


def render_batch_results(df_results: pd.DataFrame) -> None:
    st.markdown("### Batch Prediction Results")

    counts = df_results["sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [SENTIMENT_COLORS.get(c, "#888") for c in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.45)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Number of Reviews")
    ax.set_title("Sentiment Distribution in Uploaded File", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        df_results[["review", "sentiment"]].rename(columns={"review": "Review", "sentiment": "Sentiment"}),
        use_container_width=True,
        height=350,
    )

    csv_data = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="sentiment_results.csv",
        mime="text/csv",
    )

def main():

    st.title("Customer Review Sentiment Analyzer")
    st.markdown(
        """
        Enter a product review below and this application will classify it as
        Positive, Neutral, or Negative using a machine learning model
        trained on thousands of real customer reviews.
        """
    )

    model = load_model(MODEL_PATH)

    if model is None:
        st.error(
            "Model file not found. Please train the model first by running:\n\n"
            "`python train.py`\n\n"
            "from the project root directory."
        )
        st.stop()

    tab1, tab2 = st.tabs(["Single Review", "Batch Analysis"])

    with tab1:
        st.markdown("### Analyze a Single Review")

        example_reviews = [
            "Select an example...",
            "This product exceeded all my expectations. Superb build quality and fast shipping.",
            "Completely useless. Stopped working after two days. Requesting a refund.",
            "It is an average product. Works as advertised but nothing extraordinary.",
            "Good value for money but the packaging was damaged when it arrived.",
        ]

        selected = st.selectbox("Or choose an example review:", example_reviews)

        user_input = st.text_area(
            "Enter your review here:",
            value="" if selected == "Select an example..." else selected,
            height=140,
            placeholder="Type or paste a customer review here...",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

        if analyze_btn:
            if not user_input.strip():
                st.warning("Please enter a review before clicking Analyze.")
            else:
                with st.spinner("Analyzing review..."):
                    try:
                        result = predict(model, user_input)
                        render_sentiment_result(result)

                        with st.expander("See preprocessing details"):
                            st.markdown("Original text:")
                            st.code(user_input)
                            st.markdown("After cleaning and lemmatization:")
                            st.code(result["cleaned_text"])

                    except ValueError as e:
                        st.error(str(e))

    with tab2:
        st.markdown("Batch Analysis from CSV")
        st.markdown(
            "Upload a CSV file with a column named review containing review texts."
            "The model will classify each review and you can download the results."
        )

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="CSV must have a column named review",
        )

        if uploaded_file:
            try:
                df_uploaded = pd.read_csv(uploaded_file)

                if "review" not in df_uploaded.columns:
                    st.error(
                        f"CSV must contain a column named 'review'. "
                        f"Found columns: {', '.join(df_uploaded.columns.tolist())}"
                    )
                else:
                    st.success(f"File loaded: {len(df_uploaded)} reviews found.")

                    if st.button("Classify All Reviews", type="primary"):
                        progress_bar = st.progress(0)
                        sentiments = []

                        for i, review in enumerate(df_uploaded["review"].astype(str)):
                            try:
                                result = predict(model, review)
                                sentiments.append(result["sentiment"])
                            except ValueError:
                                sentiments.append("Unknown")

                            progress_bar.progress((i + 1) / len(df_uploaded))

                        df_uploaded["sentiment"] = sentiments
                        render_batch_results(df_uploaded)

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")


if __name__ == "__main__":
    main()