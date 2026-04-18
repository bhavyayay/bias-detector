# Political Bias Detector — Local Web App

A Flask web app that trains a TF-IDF + LightGBM classifier on your dataset
at startup and serves a live bias detection dashboard.

## Setup

```bash
# 1. Clone / unzip the project folder
cd bias_app

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure your dataset is in the same folder
#    The file should be named: data.csv

# 5. Run the app
python app.py
```

Then open **http://localhost:5000** in your browser.

## Project structure

```
bias_app/
├── app.py              ← Flask backend + model training
├── data.csv            ← Your political_social_media.csv (rename it)
├── requirements.txt
├── README.md
└── templates/
    └── index.html      ← Frontend dashboard
```

## Features

- **Overview tab** — dataset stats, bias distribution chart, message types, platform split
- **Live Detector** — type any text, get instant partisan/neutral verdict with confidence
- **Model Results** — confusion matrix + precision/recall/F1 auto-computed from your data
- **Pipeline** — methodology walkthrough

## API

`POST /predict`  
Body: `{"text": "your text here"}`  
Returns: `{"label": "partisan"|"neutral", "confidence": 87.3, "proba_neutral": 12.7, "proba_partisan": 87.3, "top_words": [...]}`
