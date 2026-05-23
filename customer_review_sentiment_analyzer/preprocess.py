import re
import string
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def setup_nltk_resources():
    resources = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'stopwords': 'corpora/stopwords',
        'omw-1.4': 'corpora/omw-1.4'
    }
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}...")
            nltk.download(resource, quiet=True)

# Programmatically set up resources on load
setup_nltk_resources()

try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    logger.info("NLTK preprocessing resources loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load NLTK resources: {e}")
    raise


def remove_html_tags(text: str) -> str:
    pattern = re.compile(r"<.*?>")
    return pattern.sub("", text)


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\S+", "", text)


def remove_special_characters(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s']", " ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def lemmatize_and_filter(text: str) -> str:
    tokens = word_tokenize(text)
    
    cleaned_tokens = []
    for token in tokens:
        token_lower = token.lower()
        if (token_lower not in stop_words 
                and token_lower.isalpha() 
                and len(token_lower) > 1):
            lemma = lemmatizer.lemmatize(token_lower)
            cleaned_tokens.append(lemma)
            
    return " ".join(cleaned_tokens)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)
    text = lemmatize_and_filter(text)
    return text


def map_rating_to_sentiment(rating: int) -> str:
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"


def preprocess_dataframe(df, text_col: str = "Text", rating_col: str = "Score"):
    logger.info(f"Starting preprocessing. Shape: {df.shape}")

    # Drop rows with missing values in key columns
    df = df.dropna(subset=[text_col, rating_col]).copy()
    logger.info(f"After dropping NaN rows. Shape: {df.shape}")

    # Map ratings to sentiment labels
    df["sentiment"] = df[rating_col].astype(int).apply(map_rating_to_sentiment)

    # Apply full text cleaning pipeline
    logger.info("Applying text cleaning pipeline. This may take several minutes...")
    df["clean_text"] = df[text_col].apply(clean_text)

    # Drop rows where clean_text became empty after cleaning
    df = df[df["clean_text"].str.strip() != ""]
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")

    return df[["clean_text", "sentiment"]]


if __name__ == "__main__":
    import argparse
    import pandas as pd
    import os

    parser = argparse.ArgumentParser(description="Preprocess customer reviews dataset using NLTK.")
    parser.add_argument("--input", required=True, help="Path to the raw input CSV file.")
    parser.add_argument("--output", required=True, help="Path to save the processed output CSV file.")
    parser.add_argument("--text-col", default="Text", help="Name of the review text column.")
    parser.add_argument("--rating-col", default="Score", help="Name of the rating column.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of rows to process.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        exit(1)

    logger.info(f"Loading data from {args.input}...")
    if args.limit:
        df = pd.read_csv(args.input, nrows=args.limit)
    else:
        df = pd.read_csv(args.input)

    df_processed = preprocess_dataframe(df, text_col=args.text_col, rating_col=args.rating_col)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_processed.to_csv(args.output, index=False)
    logger.info(f"Processed data saved to {args.output}")