import html
import json
import re
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import spacy
import streamlit as st
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


st.set_page_config(
    page_title="NewsBot Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "newsbot_dataset_sample.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

SELECTED_CATEGORIES = [
    "POLITICS",
    "ENTERTAINMENT",
    "BUSINESS",
    "SPORTS",
    "TECH",
    "WELLNESS",
]
TARGET_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "NORP", "EVENT"}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 196, 120, 0.22), transparent 30%),
                radial-gradient(circle at top right, rgba(118, 168, 255, 0.18), transparent 28%),
                linear-gradient(180deg, #faf6ef 0%, #f5efe4 42%, #efe6d6 100%);
            color: #1f1b16;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        .hero-card {
            background: rgba(255, 252, 246, 0.78);
            border: 1px solid rgba(96, 70, 38, 0.14);
            border-radius: 24px;
            padding: 1.35rem 1.5rem 1.25rem 1.5rem;
            box-shadow: 0 20px 50px rgba(63, 45, 24, 0.08);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            display: inline-block;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8a5b1f;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .hero-title {
            font-size: 2.25rem;
            line-height: 1.04;
            font-weight: 800;
            color: #1f1b16;
            margin-bottom: 0.45rem;
        }
        .hero-copy {
            color: #4c4137;
            font-size: 1rem;
            line-height: 1.5;
            max-width: 58rem;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(96, 70, 38, 0.14);
            border-radius: 20px;
            padding: 1rem 1rem 0.85rem 1rem;
            box-shadow: 0 18px 40px rgba(63, 45, 24, 0.06);
            height: 100%;
        }
        .metric-label {
            color: #735e49;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: #1f1b16;
            line-height: 1;
        }
        .metric-note {
            color: #5c4e41;
            font-size: 0.92rem;
            margin-top: 0.45rem;
            line-height: 1.4;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #1f1b16;
            margin-bottom: 0.25rem;
        }
        .section-copy {
            color: #5b4d3f;
            margin-bottom: 0.75rem;
        }
        .insight-chip {
            display: inline-block;
            background: rgba(138, 91, 31, 0.08);
            border: 1px solid rgba(138, 91, 31, 0.16);
            color: #71480f;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            margin: 0 0.35rem 0.45rem 0;
            font-size: 0.88rem;
            font-weight: 600;
        }
        .result-panel {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(96, 70, 38, 0.14);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 40px rgba(63, 45, 24, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_nltk_resources() -> None:
    resource_lookups = {
        "punkt": ["tokenizers/punkt", "tokenizers/punkt.zip"],
        "punkt_tab": ["tokenizers/punkt_tab", "tokenizers/punkt_tab.zip"],
        "stopwords": ["corpora/stopwords", "corpora/stopwords.zip"],
        "wordnet": ["corpora/wordnet", "corpora/wordnet.zip", "corpora/wordnet.zip/wordnet"],
        "omw-1.4": ["corpora/omw-1.4", "corpora/omw-1.4.zip"],
        "vader_lexicon": [
            "sentiment/vader_lexicon",
            "sentiment/vader_lexicon.zip",
            "sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt",
        ],
    }

    for resource_name, candidate_paths in resource_lookups.items():
        available = False
        for candidate_path in candidate_paths:
            try:
                nltk.data.find(candidate_path)
                available = True
                break
            except LookupError:
                continue

        if not available:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass


def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


@st.cache_resource(show_spinner=False)
def build_runtime_objects():
    ensure_nltk_resources()

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. Run: python -m spacy download en_core_web_sm"
        ) from exc

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    sentiment_analyzer = SentimentIntensityAnalyzer()

    def preprocess_text(text: str) -> str:
        cleaned = clean_text(text)
        tokens = word_tokenize(cleaned) if cleaned else []
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if len(token) > 2]
        return " ".join(tokens)

    df = load_dataset()
    df["processed_text"] = df["full_text"].apply(preprocess_text)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(df["processed_text"])
    classifier = MultinomialNB()
    classifier.fit(tfidf_matrix, df["category"])

    return {
        "nlp": nlp,
        "lemmatizer": lemmatizer,
        "stop_words": stop_words,
        "sentiment_analyzer": sentiment_analyzer,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "preprocess_text": preprocess_text,
    }


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["word_count"] = df["full_text"].fillna("").str.split().str.len()
    df["char_count"] = df["full_text"].fillna("").str.len()
    return df


@st.cache_data(show_spinner=False)
def load_project_outputs():
    with open(OUTPUT_DIR / "comprehensive_insights.json", "r", encoding="utf-8") as handle:
        insights = json.load(handle)
    with open(OUTPUT_DIR / "classification_report.json", "r", encoding="utf-8") as handle:
        classification_report = json.load(handle)

    model_comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    return insights, classification_report, model_comparison


@st.cache_data(show_spinner=False)
def compute_sentiment_timeline(df: pd.DataFrame):
    runtime = build_runtime_objects()
    analyzer = runtime["sentiment_analyzer"]
    sentiment_df = df.copy()
    sentiment_df["compound"] = sentiment_df["full_text"].fillna("").apply(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )
    yearly_sentiment = (
        sentiment_df.groupby(["year", "category"], as_index=False)["compound"]
        .mean()
        .sort_values(["year", "category"])
    )
    return yearly_sentiment


def analyze_article(title: str, content: str):
    runtime = build_runtime_objects()
    full_text = f"{title.strip()} {content.strip()}".strip()
    processed_text = runtime["preprocess_text"](full_text)
    features = runtime["vectorizer"].transform([processed_text])
    predicted_category = runtime["classifier"].predict(features)[0]
    probabilities = runtime["classifier"].predict_proba(features)[0]
    category_probabilities = dict(zip(runtime["classifier"].classes_, probabilities))

    sentiment = runtime["sentiment_analyzer"].polarity_scores(full_text)
    if sentiment["compound"] >= 0.05:
        sentiment_label = "positive"
    elif sentiment["compound"] <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    sentiment["sentiment_label"] = sentiment_label

    doc = runtime["nlp"](full_text)
    entities = []
    for ent in doc.ents:
        if ent.label_ not in TARGET_ENTITY_LABELS:
            continue
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
            }
        )

    return {
        "title": title,
        "predicted_category": predicted_category,
        "category_confidence": float(np.max(probabilities)),
        "category_probabilities": category_probabilities,
        "sentiment": sentiment,
        "entities": entities,
        "statistics": {
            "word_count": len(full_text.split()),
            "character_count": len(full_text),
        },
    }


def render_hero(insights: dict) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-kicker">Interactive Bonus Dashboard</div>
          <div class="hero-title">NewsBot Intelligence System</div>
          <div class="hero-copy">
            This Streamlit app turns the notebook into a browsable project demo. It combines the final
            classification results, saved visualizations, a live article analyzer, and a temporal-trends view
            so the project can be demonstrated as an interactive interface instead of only a static notebook.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Articles", f"{insights['dataset_overview']['articles']:,}", "Balanced sample used for the final project."),
        ("Categories", str(len(insights["dataset_overview"]["categories"])), "Politics, entertainment, business, sports, tech, wellness."),
        ("Best Macro F1", "0.719", "Linear SVM scored best during notebook evaluation."),
        ("Live Demo Model", insights["classification_performance"]["deployment_model"], "Used because it exposes article-level probabilities."),
    ]

    for column, (label, value, note) in zip([col1, col2, col3, col4], metrics):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{value}</div>
                  <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_dashboard(df: pd.DataFrame, insights: dict, model_comparison: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Project Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A quick executive view of the final notebook results and the strongest cross-module findings.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])
    with left:
        category_counts = (
            pd.DataFrame.from_dict(insights["dataset_overview"]["category_distribution"], orient="index", columns=["count"])
            .reset_index()
            .rename(columns={"index": "category"})
        )
        fig = px.bar(
            category_counts,
            x="category",
            y="count",
            color="category",
            title="Balanced Sample by Category",
        )
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.dataframe(
            model_comparison.style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Macro F1": "{:.3f}",
                    "Weighted F1": "{:.3f}",
                    "CV Macro F1 Mean": "{:.3f}",
                    "CV Macro F1 Std": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    insight_parts = [
        f"Most positive category: {insights['sentiment_insights']['most_positive_category']}",
        f"Most negative category: {insights['sentiment_insights']['most_negative_category']}",
        f"Highest proper noun usage: {insights['pos_insights']['highest_proper_noun_category']}",
        f"Longest average sentences: {insights['syntax_insights']['longest_average_sentences']}",
    ]
    st.markdown("".join(f'<span class="insight-chip">{item}</span>' for item in insight_parts), unsafe_allow_html=True)

    tfidf_rows = [
        {"category": category, "top_terms": ", ".join(terms)}
        for category, terms in insights["tfidf_insights"].items()
    ]
    st.dataframe(pd.DataFrame(tfidf_rows), use_container_width=True, hide_index=True)


def render_live_demo() -> None:
    st.markdown('<div class="section-title">Live NewsBot Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Paste a new article headline and short description to simulate the final interactive NewsBot system.</div>',
        unsafe_allow_html=True,
    )

    with st.form("analyzer_form"):
        title = st.text_input(
            "Headline",
            value="Startup Raises $180 Million To Expand Cybersecurity Platform",
        )
        content = st.text_area(
            "Short description / article summary",
            value=(
                "Investors backed the software company after strong enterprise demand, "
                "and executives said the new capital would support hiring, product expansion, "
                "and international growth."
            ),
            height=150,
        )
        submitted = st.form_submit_button("Analyze Article")

    if submitted:
        result = analyze_article(title, content)
        left, right = st.columns([1.05, 0.95])

        with left:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.subheader("Prediction Summary")
            st.metric("Predicted Category", result["predicted_category"])
            st.metric("Confidence", f"{result['category_confidence']:.2%}")
            probability_df = (
                pd.DataFrame(
                    [
                        {"category": category, "probability": probability}
                        for category, probability in result["category_probabilities"].items()
                    ]
                )
                .sort_values("probability", ascending=False)
            )
            fig = px.bar(
                probability_df,
                x="probability",
                y="category",
                orientation="h",
                color="category",
                title="Category Probabilities",
            )
            fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.subheader("Sentiment and Entities")
            st.write(
                f"Sentiment label: **{result['sentiment']['sentiment_label']}** "
                f"(compound score: `{result['sentiment']['compound']:.3f}`)"
            )
            st.write(
                f"Word count: **{result['statistics']['word_count']}** | "
                f"Character count: **{result['statistics']['character_count']}**"
            )

            if result["entities"]:
                entity_df = pd.DataFrame(result["entities"])
                st.dataframe(entity_df, use_container_width=True, hide_index=True)
            else:
                st.info("No major named entities were detected for the supported label set.")
            st.markdown("</div>", unsafe_allow_html=True)


def render_temporal_trends(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Temporal Trends Bonus</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">This section extends the base notebook with year-by-year trend analysis, which can support the advanced-analysis bonus.</div>',
        unsafe_allow_html=True,
    )

    yearly_counts = (
        df.groupby(["year", "category"], as_index=False)
        .size()
        .rename(columns={"size": "article_count"})
        .sort_values(["year", "category"])
    )
    yearly_sentiment = compute_sentiment_timeline(df)

    left, right = st.columns(2)
    with left:
        fig = px.line(
            yearly_counts,
            x="year",
            y="article_count",
            color="category",
            markers=True,
            title="Article Volume by Year and Category",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            yearly_sentiment,
            x="year",
            y="compound",
            color="category",
            markers=True,
            title="Average Sentiment by Year and Category",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    peak_year = yearly_counts.loc[yearly_counts["article_count"].idxmax()]
    most_positive_year = yearly_sentiment.loc[yearly_sentiment["compound"].idxmax()]
    st.markdown(
        "".join(
            [
                f'<span class="insight-chip">Peak category-year volume: {peak_year["category"]} in {int(peak_year["year"])} ({int(peak_year["article_count"])} sampled articles)</span>',
                f'<span class="insight-chip">Highest average yearly sentiment: {most_positive_year["category"]} in {int(most_positive_year["year"])} ({most_positive_year["compound"]:.3f})</span>',
            ]
        ),
        unsafe_allow_html=True,
    )


def render_data_explorer(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Filter the balanced sample by category, year, and keywords to inspect the actual articles behind the analysis.</div>',
        unsafe_allow_html=True,
    )

    categories = st.multiselect("Categories", options=SELECTED_CATEGORIES, default=SELECTED_CATEGORIES)
    years = sorted([int(year) for year in df["year"].dropna().unique().tolist()])
    selected_years = st.multiselect("Years", options=years, default=years)
    keyword = st.text_input("Keyword search", placeholder="Search title or description")

    filtered = df[df["category"].isin(categories) & df["year"].isin(selected_years)].copy()
    if keyword.strip():
        pattern = keyword.strip()
        filtered = filtered[
            filtered["title"].str.contains(pattern, case=False, na=False)
            | filtered["content"].str.contains(pattern, case=False, na=False)
        ]

    st.write(f"Showing **{len(filtered)}** matching articles.")
    st.dataframe(
        filtered[["date", "category", "title", "authors", "link"]].sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def render_visual_gallery() -> None:
    st.markdown('<div class="section-title">Saved Notebook Visuals</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">These charts were generated by the main notebook and are surfaced here for easier review during a bonus-point demo.</div>',
        unsafe_allow_html=True,
    )

    visual_files = [
        "category_distribution.png",
        "text_length_distribution.png",
        "tfidf_top_terms.png",
        "pos_patterns_by_category.png",
        "sentiment_by_category.png",
        "sentiment_label_distribution.png",
        "confusion_matrix.png",
        "entity_type_distribution.png",
        "entity_type_by_category_heatmap.png",
        "syntax_features_by_category.png",
    ]

    selected_visual = st.selectbox("Choose a notebook visualization", options=visual_files)
    st.image(str(OUTPUT_DIR / selected_visual), use_container_width=True)


def render_method_notes(insights: dict, classification_report: dict) -> None:
    st.markdown('<div class="section-title">Method Notes and Submission Angle</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Use this section to explain why the dashboard strengthens the assignment and how the bonus opportunities connect to the core notebook.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.1])
    with col1:
        st.subheader("Bonus Opportunities Covered")
        st.markdown(
            """
            - **Interactive Dashboard (5 pts):** this Streamlit app turns the notebook into a live, browsable interface.
            - **Advanced Analysis (5 pts):** the temporal trends section adds year-by-year category and sentiment analysis.
            - **Potential next step:** if you want more extra credit later, the cleanest extension would be a stronger research comparison instead of custom NER training.
            """
        )

    with col2:
        st.subheader("Classification Snapshot")
        metrics = classification_report["macro avg"]
        st.write(
            f"Macro precision: **{metrics['precision']:.3f}**  \n"
            f"Macro recall: **{metrics['recall']:.3f}**  \n"
            f"Macro F1: **{metrics['f1-score']:.3f}**"
        )

        recommendation_text = insights["business_recommendations"]
        for recommendation in recommendation_text:
            st.markdown(f"- {recommendation}")

    st.info(
        "Hosting note: once your GitHub repo is pushed, this app can be deployed on Streamlit Community Cloud by pointing it at `streamlit_app.py`."
    )


def main():
    inject_styles()
    insights, classification_report, model_comparison = load_project_outputs()
    df = load_dataset()

    st.sidebar.title("NewsBot Bonus App")
    view = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard",
            "Live Demo",
            "Temporal Trends",
            "Dataset Explorer",
            "Visual Gallery",
            "Method Notes",
        ],
    )
    st.sidebar.caption("Built as an interactive extra-credit extension for the ITAI 2373 midterm.")

    render_hero(insights)

    if view == "Dashboard":
        render_dashboard(df, insights, model_comparison)
    elif view == "Live Demo":
        render_live_demo()
    elif view == "Temporal Trends":
        render_temporal_trends(df)
    elif view == "Dataset Explorer":
        render_data_explorer(df)
    elif view == "Visual Gallery":
        render_visual_gallery()
    else:
        render_method_notes(insights, classification_report)


if __name__ == "__main__":
    main()
