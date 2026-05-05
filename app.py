import streamlit as st
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel
import torch
import torch.nn as nn

# --- Configuration --- #
MODEL_NAME = "google/flan-t5-base"
CLASSIFIER_MODEL_PATH = "hallucination_classifier.pt"
CLASSIFIER_TOKENIZER_NAME = "distilbert-base-uncased"  # ✅ FIXED
THRESHOLD = 0.6

st.set_page_config(page_title="Hallucination Detector Demo", layout="wide")
st.title("🧠 LLM Hallucination Detector Demo")
st.write("Enter a question below to see the generated answer and its hallucination probability.")

# --- Model Loading ---
@st.cache_resource
def load_flan_t5_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

@st.cache_resource
def load_classifier_model():
    # ✅ USE BASE TOKENIZER INSTEAD OF BROKEN LOCAL ONE
    clf_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_TOKENIZER_NAME)
    base_model = AutoModel.from_pretrained(CLASSIFIER_TOKENIZER_NAME)

    class HallucinationClassifier(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.bert = base_model
            self.pre_classifier = nn.Linear(768, 768)
            self.classifier = nn.Linear(768, 2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state[:, 0]
            hidden = self.pre_classifier(hidden)
            hidden = torch.relu(hidden)
            hidden = self.dropout(hidden)
            return self.classifier(hidden)

    clf_model = HallucinationClassifier(base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf_model.load_state_dict(
        torch.load(CLASSIFIER_MODEL_PATH, map_location=torch.device(device))
    )
    clf_model.to(device)
    clf_model.eval()

    return clf_tokenizer, clf_model, device


flan_t5_tokenizer, flan_t5_model, flan_t5_device = load_flan_t5_model()
clf_tokenizer, clf_model, clf_device = load_classifier_model()

# --- Prediction Functions ---
def generate_answer(question):
    prompt = f"Answer this question truthfully and to the best of your ability: {question}"
    inputs = flan_t5_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(flan_t5_device)

    outputs = flan_t5_model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=4,
        early_stopping=True
    )
    return flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict_confidence(question, answer):
    inputs = clf_tokenizer(
        question + " " + answer,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    ).to(clf_device)

    with torch.no_grad():
        logits = clf_model(inputs["input_ids"], inputs["attention_mask"])

    probs = torch.softmax(logits, dim=1)
    return probs[0][1].item()


# --- Streamlit UI ---
question_input = st.text_area("Your Question:", "What is the capital of France?")

if st.button("Generate Answer and Check for Hallucination"):
    if question_input:
        with st.spinner("Generating answer..."):
            generated_answer = generate_answer(question_input)

        st.subheader("Generated Answer:")
        st.write(generated_answer)

        with st.spinner("Checking for hallucination..."):
            confidence_score = predict_confidence(question_input, generated_answer)

        st.subheader("Hallucination Confidence:")
        st.metric(label="Confidence (0-1)", value=f"{confidence_score:.3f}")

        if confidence_score < THRESHOLD:
            st.error(f"🚨 Likely Hallucination (Confidence < {THRESHOLD})")
        else:
            st.success(f"✅ Not a Hallucination (Confidence >= {THRESHOLD})")
    else:
        st.warning("Please enter a question.")