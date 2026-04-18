import os, re, json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── global model state ─────────────────────────────────────────────────────────
model = None
vectorizer = None
stats = {}

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text), flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def train():
    global model, vectorizer, stats

    DATA_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df = df[["text", "bias", "audience", "message", "source"]].dropna(subset=["text","bias"])
    df["clean"] = df["text"].apply(clean_text)
    df["label"] = df["bias"].map({"neutral": 0, "partisan": 1})

    # dataset stats for the dashboard
    stats["total"] = len(df)
    stats["neutral_raw"] = int((df["label"] == 0).sum())
    stats["partisan_raw"] = int((df["label"] == 1).sum())
    stats["platform"] = df["source"].value_counts().to_dict()
    stats["audience"] = df["audience"].value_counts().to_dict()
    stats["message_types"] = df["message"].value_counts().to_dict()

    # oversample minority class
    majority = df[df.label == 0]
    minority = df[df.label == 1]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced = pd.concat([majority, minority_up])

    X_train, X_test, y_train, y_test = train_test_split(
        balanced["clean"], balanced["label"],
        test_size=0.2, random_state=42, stratify=balanced["label"]
    )

    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                           class_weight="balanced", random_state=42, verbose=-1)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["neutral","partisan"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    stats["accuracy"] = round(acc * 100, 1)
    stats["report"] = report
    stats["cm"] = cm
    stats["trained"] = True

    print(f"[✓] Model trained — accuracy: {stats['accuracy']}%")

# train on startup
train()

# ── routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", stats=stats)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    label = int(np.argmax(proba))
    confidence = round(float(proba[label]) * 100, 1)

    # top contributing words
    feature_names = vectorizer.get_feature_names_out()
    coef = model.feature_importances_
    vec_arr = vec.toarray()[0]
    active_idx = vec_arr.nonzero()[0]
    contributions = [(feature_names[i], float(coef[i] * vec_arr[i])) for i in active_idx]
    top_words = sorted(contributions, key=lambda x: x[1], reverse=True)[:6]

    return jsonify({
        "label": "partisan" if label == 1 else "neutral",
        "confidence": confidence,
        "proba_neutral": round(float(proba[0]) * 100, 1),
        "proba_partisan": round(float(proba[1]) * 100, 1),
        "top_words": [w for w, _ in top_words]
    })

@app.route("/stats")
def get_stats():
    return jsonify(stats)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
