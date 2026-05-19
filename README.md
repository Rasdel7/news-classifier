# News Article Classifier 📰

Automatically classifies news articles into
6 categories using NLP and Machine Learning.

## Live Demo
[Click here](https://news-classifier-nrvc2kvm3ymfzhgm9akhh6.streamlit.app)

## Categories
- 🏏 Sports
- 💻 Technology
- 🏛️ Politics
- 💰 Business
- 🏥 Health
- 🎬 Entertainment

## Features
- Classify any article instantly
- Confidence scores for all categories
- Bulk classification — multiple articles at once
- Example articles for quick testing
- Download results as CSV

## Model Details
- Algorithm: Logistic Regression
- Vectorizer: TF-IDF with bigrams
- Training: 90 Indian-context news articles

## Tools Used
- Python, Scikit-learn, Streamlit, Pandas

## How to Run Locally
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
python3 train_model.py
streamlit run app.py
