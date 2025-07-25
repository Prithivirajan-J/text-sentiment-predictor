import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.load_state_dict(torch.load("models/text_model_distilbert.pth", map_location=torch.device('cpu')))

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Labels mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_map[predicted_class]

# Example usage
while True:
    user_text = input("\nEnter text (or type 'exit'): ")
    if user_text.lower() == "exit":
        break
    sentiment = predict_sentiment(user_text)
    print(f"Predicted Sentiment: {sentiment}")
