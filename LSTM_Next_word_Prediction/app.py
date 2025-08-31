import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('LSTM_Next_word_Prediction/next_word_lstm.h5')
    with open('LSTM_Next_word_Prediction/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def predict_next_word(model, tokenizer, text, max_seq_len=14):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "Unknown"

# Streamlit UI
st.title("🎭 Shakespeare Next Word Predictor")
st.write("Enter some text and let the AI predict the next word based on Shakespeare's Hamlet!")

# Load model
model, tokenizer = load_model_and_tokenizer()

# Text input
user_input = st.text_input("Enter your text:", placeholder="Enter Barnardo and Francisco two")

# Prediction button
if st.button("🔮 Predict Next Word", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("Predicting next word..."):
            next_word = predict_next_word(model, tokenizer, user_input)
            
            st.write("### Prediction:")
            st.write(f"**{user_input}** → **{next_word}**")
            
            # Show probability distribution for top 5 words
            token_list = tokenizer.texts_to_sequences([user_input])[0]
            if len(token_list) >= 14:
                token_list = token_list[-13:]
            token_list = pad_sequences([token_list], maxlen=13, padding='pre')
            
            predicted_probs = model.predict(token_list, verbose=0)[0]
            top_5_indices = np.argsort(predicted_probs)[-5:][::-1]
            
            st.write("### Top 5 Predictions:")
            for i, idx in enumerate(top_5_indices, 1):
                word = next((w for w, i in tokenizer.word_index.items() if i == idx), "Unknown")
                prob = predicted_probs[idx] * 100
                st.write(f"{i}. **{word}** ({prob:.2f}%)")
    else:
        st.warning("⚠️ Please enter some text before predicting!")

# Add some example inputs
st.write("### Try these examples:")
examples = [
    "To be or not to",
    "Enter Barnardo and Francisco two",
    "The tragedy of Hamlet",
    "Oh what a"
]

for example in examples:
    if st.button(f"Try: '{example}'"):
        st.text_input("Enter your text:", value=example, key=f"example_{example}")
