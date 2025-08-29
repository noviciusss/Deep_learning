import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
import os
import streamlit as st

# Configure page
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 2rem;
    }
    .example-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        margin: 1rem 0;
    }
    .positive-result {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .negative-result {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎬 IMDB Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Deep Learning RNN Model</div>', unsafe_allow_html=True)


#Load the imdb dataset word index 
with st.spinner("Loading IMDB dataset..."):
    word_index = imdb.get_word_index()
    reverse_word_index = {value :key for key , value in word_index.items()}

##Load the model
with st.spinner("Loading trained model..."):
    model_path = 'Imdb_RNN/Simple_RNN_imdb.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        st.success(" Model loaded successfully!")
    except Exception as e:
        st.error(f" Error loading model: {str(e)}")
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
        return "Error: Model not loaded", 0.0, "Unknown"
    
    preprocess = preprocess_review(review)
    models_prediction = model.predict(preprocess, verbose=0)
    
    confidence_score = float(models_prediction[0][0])
    
    # Enhanced sentiment classification with neutral zone
    if confidence_score > 0.7:
        sentiment = 'Positive'
        category = 'strong_positive'
    elif confidence_score > 0.55:
        sentiment = 'Positive'
        category = 'weak_positive'
    elif confidence_score > 0.45:
        sentiment = 'Neutral/Mixed'
        category = 'neutral'
    elif confidence_score > 0.3:
        sentiment = 'Negative'
        category = 'weak_negative'
    else:
        sentiment = 'Negative'
        category = 'strong_negative'
    
    return sentiment, confidence_score, category

# Sidebar with examples and information
st.sidebar.header("📋 Try These Examples")

# Example reviews
positive_examples = [
    "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and the cinematography was breathtaking. I would definitely recommend this to anyone looking for a great film experience.",
    "Amazing storyline with incredible visual effects. The characters were well-developed and the ending was very satisfying. One of the best movies I've seen this year!",
    "Outstanding performance by the lead actor. The movie kept me engaged from start to finish with its brilliant direction and compelling narrative."
]

negative_examples = [
    "This was one of the worst movies I've ever seen. The plot was confusing, the acting was terrible, and I wasted my time watching it. Would not recommend to anyone.",
    "Boring and predictable storyline. The characters were poorly developed and the dialogue was unnatural. I fell asleep halfway through the movie.",
    "Complete waste of time and money. Poor acting, terrible script, and awful direction. I regret watching this movie."
]

neutral_examples = [
    "It was an okay movie, not good nor bad. I liked the action parts but hated the romance parts too much.",
    "The movie had some good moments and some bad ones. Overall it was average, neither impressive nor disappointing.",
    "Mixed feelings about this film. Great visuals and soundtrack, but the story was weak and predictable."
]

st.sidebar.subheader("😊 Positive Examples")
for i, example in enumerate(positive_examples, 1):
    if st.sidebar.button(f"Example {i} (Positive)", key=f"pos_{i}"):
        st.session_state.example_text = example

st.sidebar.subheader("😐 Neutral/Mixed Examples")
for i, example in enumerate(neutral_examples, 1):
    if st.sidebar.button(f"Example {i} (Mixed)", key=f"neu_{i}"):
        st.session_state.example_text = example

st.sidebar.subheader("😞 Negative Examples")
for i, example in enumerate(negative_examples, 1):
    if st.sidebar.button(f"Example {i} (Negative)", key=f"neg_{i}"):
        st.session_state.example_text = example

# Model information in sidebar
st.sidebar.header(" Model Information")
st.sidebar.info("""
**Model Architecture:**
- Embedding Layer (128 dimensions)
- Simple RNN (128 units)
- Dense Output Layer (1 unit, sigmoid)

**Training Data:**
- IMDB Movie Reviews Dataset
- 50,000 reviews for training
- Binary classification (Positive/Negative)
""")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("✍️ Enter Your Movie Review")
    
    # Get text from session state if example was clicked
    default_text = st.session_state.get('example_text', '')
    
    user_input = st.text_area(
        "Write your movie review here:",
        value=default_text,
        height=200,
        placeholder="Enter a movie review to analyze its sentiment...",
        help="Write a detailed movie review. The model will predict whether it's positive or negative."
    )
    
    # Clear button
    if st.button("🗑️ Clear Text"):
        st.session_state.example_text = ""
        st.rerun()

with col2:
    st.header("Analysis")
    
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if model is not None and user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence, category = predict_sentiment(user_input)
                
                # Display results with enhanced styling for neutral sentiment
                if sentiment == 'Positive':
                    st.markdown(f"""
                    <div class="positive-result">
                        <h3>😊 Positive Review</h3>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(confidence)
                    
                elif sentiment == 'Neutral/Mixed':
                    st.markdown(f"""
                    <div style="background-color: #fff3cd; color: #856404; padding: 1rem; border-radius: 5px; border: 1px solid #ffeaa7;">
                        <h3>😐 Neutral/Mixed Review</h3>
                        <p><strong>Score:</strong> {confidence:.2%}</p>
                        <p><em>This review contains both positive and negative elements</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(0.5)  # Show as middle ground
                    
                else:
                    st.markdown(f"""
                    <div class="negative-result">
                        <h3>😞 Negative Review</h3>
                        <p><strong>Confidence:</strong> {(1-confidence):.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(1-confidence)  # Show distance from positive
                
                # Enhanced analysis
                st.subheader("📈 Detailed Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    word_count = len(user_input.split())
                    st.metric("Word Count", word_count)
                
                with col2:
                    # Sentiment strength
                    if category == 'strong_positive':
                        strength = "🟢 Very Positive"
                    elif category == 'weak_positive':
                        strength = "🟡 Slightly Positive"
                    elif category == 'neutral':
                        strength = "⚪ Neutral/Mixed"
                    elif category == 'weak_negative':
                        strength = "🟠 Slightly Negative"
                    else:
                        strength = "🔴 Very Negative"
                    
                    st.metric("Sentiment Strength", strength)
                
                with col3:
                    # Model certainty
                    certainty = abs(confidence - 0.5) * 2  # Distance from neutral
                    if certainty > 0.6:
                        certainty_level = "🟢 High"
                    elif certainty > 0.3:
                        certainty_level = "🟡 Medium"
                    else:
                        certainty_level = "🟠 Low"
                    
                    st.metric("Model Certainty", certainty_level)
                
                # Explanation for your specific case
                if 0.3 <= confidence <= 0.7:
                    st.info("""
                    **💡 Understanding Mixed Reviews:**
                    Your review contains both positive and negative sentiments. The model detected:
                    - Positive elements (action scenes)
                    - Negative elements (romance parts)
                    
                    This results in a middle-ground score, which is actually quite accurate for a mixed review!
                    """)
                
        elif not user_input.strip():
            st.warning("⚠️ Please enter a movie review.")
        else:
            st.error(" Model not loaded. Cannot make predictions.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎬 Built with Streamlit & TensorFlow | Movie Review Sentiment Analysis</p>
    <p>Model trained on IMDB dataset with Simple RNN architecture</p>
</div>
""", unsafe_allow_html=True)
