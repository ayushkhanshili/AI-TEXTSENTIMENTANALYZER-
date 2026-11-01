  

import streamlit as st
import torch
import re
import pandas as pd
import plotly.express as px

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# -----------------------------
# CUSTOM DESIGN
# -----------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #172554);
    color: white;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #60A5FA;
}
.subtitle {
    text-align: center;
    color: #CBD5E1;
    font-size: 17px;
}
.result-card {
    background: #1E293B;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD AI MODEL
# -----------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------

def preprocess_text(text):

    text = text.strip()

    text = re.sub(
        r'@\w+',
        '@user',
        text
    )

    text = re.sub(
        r'https?://\S+|www\.\S+',
        'http',
        text
    )

    return text


# -----------------------------
# SENTIMENT PREDICTION
# -----------------------------

def analyze_sentiment(text, tokenizer, model):

    text = preprocess_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    results = {}

    for index, score in enumerate(probabilities):

        label = model.config.id2label[index].lower()

        results[label] = float(score) * 100

    prediction = max(
        results,
        key=results.get
    )

    return prediction, results


# -----------------------------
# MAIN WEBSITE
# -----------------------------

st.markdown(
    '<div class="main-title">🤖 AI TEXT SENTIMENT ANALYZER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Understand emotions using Artificial Intelligence</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "This AI analyzes text and predicts whether "
    "the sentiment is Positive, Negative or Neutral."
)

# -----------------------------
# INPUT SECTION
# -----------------------------

st.subheader("📝 Enter Your Text")

text_input = st.text_area(
    "Write your review, comment or message:",
    height=150,
    placeholder="Example: This product is amazing! I really love it."
)

st.caption("Supports English and Hindi. Hinglish can also be tested.")

analyze_button = st.button(
    "🚀 Analyze Sentiment",
    use_container_width=True,
    type="primary"
)

# -----------------------------
# ANALYSIS SECTION
# -----------------------------

if analyze_button:

    if not text_input.strip():

        st.warning("Please enter some text first.")

    elif len(text_input) > 10000:

        st.warning("Please keep your text under 10,000 characters.")

    else:

        try:

            with st.spinner("AI is analyzing your text..."):

                tokenizer, model = load_model()

                prediction, scores = analyze_sentiment(
                    text_input,
                    tokenizer,
                    model
                )

            st.divider()

            st.subheader("📊 Analysis Results")

            if prediction == "positive":

                emoji = "😊"
                title = "POSITIVE SENTIMENT"
                color = "#22C55E"

            elif prediction == "negative":

                emoji = "😡"
                title = "NEGATIVE SENTIMENT"
                color = "#EF4444"

            else:

                emoji = "😐"
                title = "NEUTRAL SENTIMENT"
                color = "#FACC15"

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h1>{emoji}</h1>
                        <h2 style="color:{color}">
                            {title}
                        </h2>
                        <h3>
                            Confidence: {scores[prediction]:.2f}%
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:

                st.markdown("### 📈 Sentiment Probabilities")

                for label in ["positive", "neutral", "negative"]:

                    score = scores[label]

                    st.write(
                        f"**{label.capitalize()} — {score:.2f}%**"
                    )

                    st.progress(
                        int(round(score))
                    )

            # -----------------------------
            # CHART
            # -----------------------------

            st.divider()

            st.subheader("📉 Sentiment Distribution")

            df = pd.DataFrame({
                "Sentiment": [
                    "Positive",
                    "Neutral",
                    "Negative"
                ],
                "Probability": [
                    scores["positive"],
                    scores["neutral"],
                    scores["negative"]
                ]
            })

            fig = px.bar(
                df,
                x="Sentiment",
                y="Probability",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#22C55E",
                    "Neutral": "#FACC15",
                    "Negative": "#EF4444"
                },
                range_y=[0, 100],
                text_auto=".2f"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#101827",
                plot_bgcolor="#101827",
                font_color="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -----------------------------
            # EXPLANATION
            # -----------------------------

            st.subheader("💡 AI Interpretation")

            st.write(
                f"The AI model classified your text as "
                f"**{prediction.upper()}** sentiment."
            )

            st.caption(
                "Confidence is the model's predicted probability, "
                "not a guarantee of correctness. Sarcasm and "
                "mixed-language text may be difficult to classify."
            )

            # -----------------------------
            # DOWNLOAD REPORT
            # -----------------------------

            report = f"""
AI TEXT SENTIMENT ANALYSIS REPORT

Input Text:
{text_input}

Predicted Sentiment:
{prediction.upper()}

Positive: {scores['positive']:.2f}%
Neutral: {scores['neutral']:.2f}%
Negative: {scores['negative']:.2f}%
"""

            st.download_button(
                "📥 Download Analysis Report",
                data=report,
                file_name="sentiment_report.txt",
                mime="text/plain"
            )

        except Exception as error:

            st.error(
                "Something went wrong while loading the AI model."
            )

            st.caption(str(error))


# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.markdown()
   
   
