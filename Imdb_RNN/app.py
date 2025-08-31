import numpy as np
import os
# Fix TensorFlow compatibility issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
import streamlit as st


#Load the imdb dataset word index 
word_index = imdb.get_word_index()
reverse_word_index = {value :key for key , value in word_index.items()}

##Load the model
#model_path = 'Simple_RNN_imdb.h5'
model_path = 'Imdb_RNN/Simple_RNN_imdb.h5'
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.write("Available files in root:", os.listdir('.'))
    if os.path.exists('Imdb_RNN'):
        st.write("Available files in Imdb_RNN:", os.listdir('Imdb_RNN'))
    model = None

###Step 2-helper Function
def decode_review(endcoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in endcoded_review])

def preprocess_review(text):
    # Clean the text: remove punctuation and convert to lowercase
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    words = text.lower().split()
    
    # Filter out empty strings and encode words
    words = [word for word in words if word.strip()]
    endoded_review = [word_index.get(word, 2) + 3 for word in words]
    
    # Handle empty reviews
    if not endoded_review:
        endoded_review = [2 + 3]  # Default unknown word if no valid words found
    
    padded_review = sequence.pad_sequences([endoded_review], maxlen=2500)
    return padded_review

###Predicton function

def predict_sentiment(review):
    if model is None:
        return "Error: Model not loaded", 0.0
    
    try:
        preprocess = preprocess_review(review)
        models_prediction = model.predict(preprocess)

        sentiment = 'Positive' if models_prediction[0][0] > 0.65 else ('Neutral' if models_prediction[0][0] > 0.45 else 'Negative')
        return sentiment, float(models_prediction[0][0])
    except Exception as e:
        return f"Error during prediction: {str(e)}", 0.0

import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (positive or negative).")

user_input = st.text_area("Movie Review", height=200)

if st.button("Predict Sentiment"):
    if model is None:
        st.error("Cannot make prediction: Model failed to load. Please check if the model file exists.")
    elif not user_input.strip():
        st.warning("Please enter a movie review to analyze.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")
