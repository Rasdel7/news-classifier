import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="wide"
)

st.title("📰 News Article Classifier")
st.markdown("Automatically categorize any news "
            "article into Sports, Tech, Politics "
            "and more.")
st.markdown("---")

# Category colors and emojis
CAT_CONFIG = {
    'Sports':        {'color': '#e74c3c', 'emoji': '🏏'},
    'Technology':    {'color': '#3498db', 'emoji': '💻'},
    'Politics':      {'color': '#9b59b6', 'emoji': '🏛️'},
    'Business':      {'color': '#f39c12', 'emoji': '💰'},
    'Health':        {'color': '#2ecc71', 'emoji': '🏥'},
    'Entertainment': {'color': '#1abc9c', 'emoji': '🎬'}
}

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl',      'rb') as f:
        model      = pickle.load(f)
    with open('tfidf.pkl',      'rb') as f:
        tfidf      = pickle.load(f)
    with open('categories.pkl', 'rb') as f:
        categories = pickle.load(f)
    return model, tfidf, categories

try:
    model, tfidf, categories = load_model()
    st.success("✅ Model loaded — "
               "6 categories, trained on "
               "90 articles")
except:
    st.error("Run train_model.py first!")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📰 Classify Article",
    "📊 Bulk Classify",
    "📈 Model Info"
])

# Tab 1 — Single
with tab1:
    st.markdown("### Classify Any News Article")

    # Example articles
    examples = {
        "Select an example...": "",
        "Cricket Match": "India defeated Australia by 5 wickets in the thrilling T20 match. Virat Kohli scored 85 runs off 52 balls to lead the team to victory.",
        "AI Launch": "OpenAI released its most powerful language model today with breakthrough reasoning capabilities that surpass human performance on multiple benchmarks.",
        "Election News": "The ruling party won the state assembly elections with a two-thirds majority, defeating the opposition in 180 out of 234 constituencies.",
        "Stock Market": "The Sensex surged 1200 points today driven by strong quarterly earnings from IT companies and positive global market sentiment.",
        "Health Study": "A new study published in The Lancet found that regular exercise for 30 minutes daily reduces risk of heart disease by 40 percent.",
        "Bollywood": "The new Bollywood blockbuster crossed 300 crore rupees in its opening weekend becoming the highest grossing film of the year."
    }

    example_choice = st.selectbox(
        "Load an example:",
        list(examples.keys())
    )

    article_text = st.text_area(
        "Paste article text here:",
        value=examples[example_choice],
        height=200,
        placeholder="Paste any news article..."
    )

    if st.button("🔍 Classify Article",
                 type="primary"):
        if article_text.strip():
            vectorized  = tfidf.transform(
                [article_text])
            prediction  = model.predict(
                vectorized)[0]
            probabilities = model.predict_proba(
                vectorized)[0]

            config = CAT_CONFIG.get(
                prediction,
                {'color': '#95a5a6',
                 'emoji': '📰'})

            st.markdown("---")
            st.markdown(
                f"<h2 style='text-align:center;"
                f"color:{config['color']}'>"
                f"{config['emoji']} {prediction}"
                f"</h2>",
                unsafe_allow_html=True
            )

            # Probability bars
            st.markdown("### 📊 Confidence Scores")
            prob_df = pd.DataFrame({
                'Category':    categories,
                'Probability': probabilities
            }).sort_values(
                'Probability', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 5))
            bar_colors = [
                CAT_CONFIG.get(
                    cat,
                    {'color': '#95a5a6'}
                )['color']
                for cat in prob_df['Category']
            ]
            bars = ax.barh(
                prob_df['Category'],
                prob_df['Probability'] * 100,
                color=bar_colors,
                edgecolor='black'
            )
            for bar, val in zip(
                bars,
                prob_df['Probability'] * 100
            ):
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() +
                    bar.get_height()/2,
                    f'{val:.1f}%',
                    va='center',
                    fontweight='bold'
                )
            ax.set_xlim(0, 115)
            ax.set_title(
                'Classification Confidence',
                fontsize=13)
            ax.set_xlabel('Probability (%)')
            plt.tight_layout()
            st.pyplot(fig)

            # Top prediction details
            top_prob = prob_df.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Category",   prediction)
            c2.metric("Confidence",
                      f"{top_prob['Probability']*100:.1f}%")
            c3.metric("Word Count",
                      len(article_text.split()))
        else:
            st.warning(
                "Please enter article text!")

# Tab 2 — Bulk
with tab2:
    st.markdown("### Classify Multiple Articles")
    st.markdown(
        "Enter one article headline "
        "or text per line.")

    bulk_input = st.text_area(
        "Enter articles (one per line):",
        placeholder=(
            "India won cricket world cup\n"
            "Apple launched new iPhone model\n"
            "Parliament passed education bill\n"
            "Sensex hit all time high today\n"
            "New cancer vaccine shows promise\n"
            "Bollywood film breaks box office record"
        ),
        height=200
    )

    if st.button("📊 Classify All",
                 type="primary"):
        if bulk_input.strip():
            lines = [
                l.strip()
                for l in bulk_input.split('\n')
                if l.strip()
            ]

            results = []
            for line in lines:
                vec   = tfidf.transform([line])
                pred  = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]
                conf  = max(proba) * 100
                emoji = CAT_CONFIG.get(
                    pred,
                    {'emoji': '📰'})['emoji']
                results.append({
                    'Article':    line[:60] +
                                  '...' if
                                  len(line) > 60
                                  else line,
                    'Category':   f"{emoji} {pred}",
                    'Confidence': f"{conf:.1f}%"
                })

            df = pd.DataFrame(results)
            st.dataframe(df,
                         use_container_width=True,
                         hide_index=True)

            # Category distribution
            cat_counts = pd.Series(
                [r['Category']
                 for r in results]
            ).value_counts()

            fig2, ax2 = plt.subplots(
                figsize=(8, 4))
            colors2   = [
                CAT_CONFIG.get(
                    cat.split(' ')[-1],
                    {'color': '#95a5a6'}
                )['color']
                for cat in cat_counts.index
            ]
            ax2.bar(cat_counts.index,
                    cat_counts.values,
                    color=colors2,
                    edgecolor='black')
            ax2.set_title(
                'Category Distribution',
                fontsize=13)
            ax2.set_ylabel('Count')
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig2)

            st.download_button(
                "⬇️ Download Results",
                df.to_csv(index=False),
                "classified_articles.csv",
                "text/csv"
            )
        else:
            st.warning(
                "Please enter some articles!")

# Tab 3 — Model Info
with tab3:
    st.markdown("### 📈 Model Performance")

    charts = [
        'model_comparison.png',
        'confusion_matrix.png'
    ]
    for chart in charts:
        if os.path.exists(chart):
            st.image(chart,
                     use_column_width=True)
        else:
            st.info(
                f"Run train_model.py "
                f"to generate {chart}")

    st.markdown("### 📋 Categories")
    cat_df = pd.DataFrame([{
        'Category': cat,
        'Emoji':    CAT_CONFIG.get(
            cat, {'emoji': '📰'})['emoji'],
        'Examples': {
            'Sports':
                'Cricket, Football, Tennis',
            'Technology':
                'AI, Smartphones, Space',
            'Politics':
                'Elections, Parliament, Policy',
            'Business':
                'Stock Market, Companies, Economy',
            'Health':
                'Medicine, Fitness, Research',
            'Entertainment':
                'Bollywood, Music, OTT'
        }.get(cat, 'Various')
    } for cat in categories])
    st.dataframe(cat_df,
                 use_container_width=True,
                 hide_index=True)

    info = pd.DataFrame({
        'Detail': [
            'Algorithm',
            'Vectorizer',
            'Categories',
            'Training articles',
            'Features'
        ],
        'Value': [
            'Logistic Regression',
            'TF-IDF (bigrams)',
            '6',
            '90',
            '5,000'
        ]
    })
    st.dataframe(info,
                 use_container_width=True,
                 hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "News Article Classifier | "
    "6 categories, instant classification"
)