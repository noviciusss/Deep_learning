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
try:
    model = tf.keras.models.load_model('Simple_RNN_imdb.h5')
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.info("This is a TensorFlow version compatibility issue.")
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
    preprocess = preprocess_review(review)
    models_prediction = model.predict(preprocess)
    
    sentiment = 'Positive' if models_prediction[0][0]>0.5 else 'Negative'
    return sentiment, float(models_prediction[0][0])

import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (positive or negative).")

user_input = st.text_area("Movie Review", height=200)

if st.button("Predict Sentiment"):
    sentiment, confidence = predict_sentiment(user_input)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.2f}")