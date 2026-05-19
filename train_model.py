import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(
    os.path.abspath(__file__)))

# Create training dataset
print("Creating news dataset...")

news_data = {
    'text': [
        # Sports
        "India won the cricket world cup defeating Australia in the final",
        "Virat Kohli scored a brilliant century in the test match",
        "Manchester United signed a new striker for 80 million pounds",
        "The Olympics will be held in Paris this summer with 10000 athletes",
        "Federer announced retirement after 20 grand slam titles",
        "IPL auction saw record breaking bids for top players this season",
        "Neymar transferred to Al Hilal for record breaking fee",
        "The FIFA World Cup final drew 1 billion viewers worldwide",
        "Lionel Messi won his eighth Ballon dOr award last night",
        "Lewis Hamilton won the Formula 1 championship for eighth time",
        "India beat Pakistan in the Asia Cup final by 6 wickets",
        "Rohit Sharma became captain of Indian cricket team",
        "Novak Djokovic won Wimbledon for the seventh time",
        "LeBron James scored his 40000th NBA career point",
        "The Indian football team qualified for Asian Cup",

        # Technology
        "Apple released the new iPhone 16 with advanced AI features",
        "Google launched its new AI model beating GPT-4 on benchmarks",
        "Tesla unveiled its fully autonomous self driving robotaxi",
        "Microsoft acquired a gaming company for 69 billion dollars",
        "Meta released new virtual reality headset for consumers",
        "OpenAI launched GPT-5 with multimodal capabilities",
        "Amazon Web Services launched new quantum computing service",
        "Samsung released foldable phone with revolutionary display technology",
        "ISRO successfully launched satellite using new rocket technology",
        "SpaceX launched 60 more Starlink satellites into orbit",
        "Intel released new processor with 3 nanometer architecture",
        "Nvidia GPU demand surges due to AI training requirements",
        "Reliance Jio launched 5G services across all major Indian cities",
        "Wipro and Infosys announced major AI transformation initiatives",
        "GitHub Copilot reached 1 million paid subscribers milestone",

        # Politics
        "Prime Minister Modi announced new economic policy for farmers",
        "Parliament passed new education bill with major reforms",
        "The US elections saw record voter turnout across all states",
        "United Nations security council discussed climate change resolution",
        "G20 summit in India focused on global economic recovery",
        "Opposition party won state elections by large majority",
        "New foreign policy announced strengthening bilateral ties",
        "Supreme Court delivered landmark judgment on privacy rights",
        "Government launched new scheme for rural development",
        "The budget session of parliament began with heated debates",
        "President signed new healthcare reform bill into law",
        "Governor appointed new chief minister after coalition talks",
        "India and China held border talks to reduce tensions",
        "Election commission announced dates for upcoming state polls",
        "New anti corruption law passed after years of debate",

        # Business
        "Reliance Industries reported record quarterly profits this year",
        "Stock market reached all time high driven by IT sector rally",
        "RBI announced interest rate cut to boost economic growth",
        "Adani Group expanded into new renewable energy sectors",
        "Startup ecosystem in India raised 10 billion dollars in funding",
        "Tata Motors launched new electric vehicle with 500km range",
        "Flipkart and Amazon compete for festive season sales dominance",
        "Gold prices hit record high amid global economic uncertainty",
        "Sensex crossed 80000 points for the first time in history",
        "New GST reforms expected to boost manufacturing sector growth",
        "Zomato and Swiggy report growth in tier 2 city food delivery",
        "HDFC Bank merged with HDFC Limited in landmark deal",
        "India became third largest economy overtaking Japan this year",
        "Foreign direct investment in India grew 20 percent this quarter",
        "UPI transactions crossed 10 billion monthly milestone",

        # Health
        "New vaccine developed against malaria showing 90 percent efficacy",
        "WHO declared end of COVID pandemic after three years",
        "AIIMS developed new cancer treatment using gene therapy",
        "Yoga and meditation proven to reduce stress study finds",
        "New diabetes drug reduces risk of heart disease significantly",
        "Government launched free health insurance scheme for poor families",
        "Mental health awareness campaign launched across Indian schools",
        "Scientists discovered new antibiotic effective against superbugs",
        "Apollo Hospitals expanded telemedicine services to rural areas",
        "Dengue cases rise in several states health ministry issues alert",
        "New blood test can detect cancer 5 years before symptoms appear",
        "India eradicated polio becoming third country to achieve this",
        "Ayushman Bharat scheme covered 50 million families this year",
        "Sleep deprivation linked to increased Alzheimer risk new study",
        "Exercise shown to be as effective as antidepressants new research",

        # Entertainment
        "Bollywood blockbuster crossed 500 crore rupees at box office",
        "Netflix India original series won international Emmy award",
        "AR Rahman composed music for Hollywood blockbuster film",
        "Deepika Padukone starred in most expensive Indian film ever",
        "OTT platforms saw record subscriptions during festive season",
        "Shah Rukh Khan film broke opening day box office record",
        "Indian music artist reached 1 billion streams on Spotify",
        "Cannes film festival selected Indian documentary for competition",
        "YouTube India celebrated 100 million local creators milestone",
        "Grammy nominated Indian classical musician performed globally",
        "RRR won Golden Globe for best non English language film",
        "Alia Bhatt won best actress award at national film awards",
        "Indian stand up comedy scene grew massively with new talent",
        "Amazon Prime original show based on Indian mythology trending",
        "Music reality show launched new singing talent across country"
    ],
    'category': (
        ['Sports'] * 15 +
        ['Technology'] * 15 +
        ['Politics'] * 15 +
        ['Business'] * 15 +
        ['Health'] * 15 +
        ['Entertainment'] * 15
    )
}

df = pd.DataFrame(news_data)
print(f"Dataset shape: {df.shape}")
print(f"\nCategory distribution:")
print(df['category'].value_counts())

# Features
X = df['text']
y = df['category']

# Split
X_train, X_test, y_train, y_test = \
    train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# Vectorize
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# Train models
models = {
    'Logistic Regression':
        LogisticRegression(max_iter=1000),
    'Naive Bayes':
        MultinomialNB(),
    'Random Forest':
        RandomForestClassifier(
            n_estimators=100,
            random_state=42)
}

results = {}
print("\nTraining models...")
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc   = accuracy_score(y_test, preds)
    results[name] = {
        'model': model,
        'preds': preds,
        'acc':   round(acc * 100, 2)
    }
    print(f"\n{name}: {acc*100:.2f}%")
    print(classification_report(
        y_test, preds))

best_name = max(
    results, key=lambda x: results[x]['acc'])
best      = results[best_name]
print(f"\nBest: {best_name} ({best['acc']}%)")

# Plot 1 — Model comparison
fig, ax = plt.subplots(figsize=(10, 5))
names  = list(results.keys())
accs   = [results[n]['acc'] for n in names]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars   = ax.bar(names, accs,
                color=colors,
                edgecolor='black')
for bar, val in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.3,
        f'{val}%',
        ha='center',
        fontweight='bold'
    )
ax.set_title(
    'Model Comparison — News Classifier',
    fontsize=14)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nModel comparison saved!")

# Plot 2 — Confusion Matrix
categories = sorted(y.unique())
cm         = confusion_matrix(
    y_test, best['preds'],
    labels=categories)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=categories,
            yticklabels=categories)
plt.title(
    f'Confusion Matrix — {best_name}',
    fontsize=13)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved!")

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(best['model'], f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('categories.pkl', 'wb') as f:
    pickle.dump(categories, f)

print(f"\nAll saved! Best: {best_name}")
print("Run app.py next!")