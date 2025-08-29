import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
import os
import streamlit as st


#Load the imdb dataset word index 
word_index = imdb.get_word_index()
reverse_word_index = {value :key for key , value in word_index.items()}

##Load the model
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
    words = text.lower().split()
    endoded_review =[word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([endoded_review], maxlen=500)
    return padded_review

###Predicton function
def predict_sentiment(review):
    if model is None:
        return "Error: Model not loaded", 0.0
    
    preprocess = preprocess_review(review)
    models_prediction = model.predict(preprocess, verbose=0)
    
    sentiment = 'Positive' if models_prediction[0][0]>0.5 else 'Negative'
    return sentiment, float(models_prediction[0][0])

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (positive or negative).")

user_input = st.text_area("Movie Review", height=200)

if st.button("Predict Sentiment"):
    if model is not None and user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f"**Sentiment**: {sentiment}")
        st.write(f"**Confidence**: {confidence:.2%}")
    elif not user_input.strip():
        st.warning("Please enter a movie review.")
    else:
        st.error("Model not loaded. Cannot make predictions.")
else:
    st.write("Please enter a movie review and click 'Predict Sentiment'.")

