import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.load_state_dict(torch.load("models/text_model_distilbert.pth", map_location=torch.device('cpu')))
model.eval()

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).tolist()
    return [label_map[p] for p in preds]

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

st.markdown("""
# 🧠 Sentiment Analyzer
Type or paste your sentence(s) below to predict sentiment using a fine-tuned DistilBERT model.
""")

st.write("🔹 **Supports multiple sentences** (enter each sentence on a new line).")
st.write("---")

user_input = st.text_area("📝 Enter text(s) below:", height=200, placeholder="Example:\nI love this product!\nThe experience was terrible.")

if st.button("🚀 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter at least one sentence.")
    else:
        st.write("## 🎯 Predictions:")
        text_list = [line.strip() for line in user_input.strip().split("\n") if line.strip()]
        predictions = predict_sentiment(text_list)
        for text, sentiment in zip(text_list, predictions):
            color = {"Positive": "green", "Neutral": "gray", "Negative": "red"}[sentiment]
            st.markdown(f"""
            <div style='border:1px solid #ddd; border-radius:8px; padding:10px; margin-bottom:10px;'>
            <strong>{text}</strong><br>
            <span style='color:{color}; font-size:20px;'>{sentiment}</span>
            </div>
            """, unsafe_allow_html=True)
