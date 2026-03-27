import streamlit as st
from transformers import pipeline
    
st.title("AG News Classifier")
st.subheader("Fine-tuned BERT Model")

@st.cache_resource
def load_model():
    
    model_path = r"C:\Users\Ahsan\Downloads\Ag_news\my_bert_model" 
    return pipeline("text-classification", model=model_path)


classifier = load_model()
labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

text_input = st.text_area("Enter News Headline + Description:", "  ")

if st.button("Classify"):
    prediction = classifier(text_input) 
    result = prediction[0] 
    
    label_str = result['label']
    label_idx = int(label_str.split('_')[-1]) if '_' in label_str else int(label_str)
    
    st.write(f"### Category: {labels[label_idx]}")
    st.write(f"**Confidence Score:** {result['score']:.4f}")
