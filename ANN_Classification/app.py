import sys, streamlit as st
st.write("Interpreter:", sys.executable)
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

##Load the trained model, scaler,onehot
model = load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot_enc = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

##Streamlit ui
st.title("Customer Churn Prediction")

## User Input
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0,)
creditscore = st.number_input('Credit Score', min_value=0, max_value=1000, value=400)
estimatedsalary = st.number_input('Estimated Salary', min_value=0,)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [1, 0])
is_active_member = st.selectbox('Is Active Member', [1, 0])

###Prepare the input
input_data= pd.DataFrame({
    'CreditScore': [creditscore],
    'Age': [age],
    'Balance': [balance],
    'EstimatedSalary': [estimatedsalary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]]
})

##One hot encode 'geo'
geo_encoded = onehot_enc.transform(input_data[['Geography']]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_enc.get_feature_names_out(['Geography']))

##Combine one hot
input_data = pd.concat([input_data.reset_index(drop=True).drop(columns=['Geography']), geo_df], axis=1)

##Input scaled
input_data_scaled = scaler.transform(input_data)

prediction_prob = model.predict(input_data_scaled)[0][0]

if prediction_prob > 0.5:
    st.success("The customer is likely to churn.")
else:
    st.success("The customer is unlikely to churn.")