import os
import sys
import pickle
import logging

from flask import Flask, request, jsonify

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from preprocess import clean_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sentiment_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    model = None
    logger.error(f"Model not found at {MODEL_PATH}. Train the model first.")


def predict_single(text: str) -> dict:
    """Run inference on a single review string."""
    if not text or not text.strip():
        raise ValueError("Review text cannot be empty.")

    cleaned = clean_text(text)
    if not cleaned.strip():
        raise ValueError("Review text is empty after preprocessing.")

    prediction = model.predict([cleaned])[0]
    result = {"review": text, "sentiment": prediction}

    try:
        proba = model.predict_proba([cleaned])[0]
        result["confidence"] = {
            cls: round(float(prob), 4)
            for cls, prob in zip(model.classes_, proba)
        }
    except AttributeError:
        pass

    return result


@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded."}), 503
    return jsonify({"status": "ok", "message": "Model is ready."}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run training first."}), 503

    data = request.get_json(silent=True)
    if not data or "review" not in data:
        return jsonify({"error": "Request body must contain a 'review' key."}), 400

    try:
        result = predict_single(data["review"])
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error."}), 500


@app.route("/batch", methods=["POST"])
def batch_predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(silent=True)
    if not data or "reviews" not in data:
        return jsonify({"error": "Request body must contain a 'reviews' list."}), 400

    reviews = data["reviews"]
    if not isinstance(reviews, list) or len(reviews) == 0:
        return jsonify({"error": "'reviews' must be a non-empty list."}), 400

    if len(reviews) > 1000:
        return jsonify({"error": "Maximum 1000 reviews per batch request."}), 400

    results = []
    for review in reviews:
        try:
            result = predict_single(str(review))
            results.append(result)
        except ValueError as e:
            results.append({"review": review, "error": str(e)})

    return jsonify({"results": results, "total": len(results)}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)