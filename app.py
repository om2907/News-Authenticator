import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import time
import random





LR_model = pickle.load(open('LR_model.pkl','rb'))
DT_model = pickle.load(open('DT_model.pkl','rb'))
GB_model = pickle.load(open('GB_model.pkl','rb'))
RF_model = pickle.load(open('RF_model.pkl','rb'))




# Define the wordopt function for preprocessing
def wordopt(text):
    if text is None:
        return ''
    else:
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W", " ", text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>,+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

# Streamlit app
st.title('Fake News Detector')

method = ['Random Forest (Recommended)','Logistic Regression', 'Decision Tree', 'Gradient Boosting' ]

selected_method = st.selectbox('Select Algorithm', method)

# Input text area for manual testing
news = st.text_area('Enter the news article:', '', height=250)

# Load the TfidfVectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

if st.button('Test'):
    
    
        # Preprocess the input news article
        processed_news = wordopt(news)
        
        # Vectorize the preprocessed news article if it's not None
        if processed_news and vectorizer:
            vectorized_news = vectorizer.transform([processed_news])
        else:
            vectorized_news = None

        # Make predictions using the trained models and display the results
        if vectorized_news is not None:
            if selected_method == 'Logistic Regression':
             predi = LR_model.predict(vectorized_news)
            elif selected_method == 'Decision Tree':
             predi = DT_model.predict(vectorized_news)
            elif selected_method == 'Gradient Boosting':
             predi = GB_model.predict(vectorized_news)
            elif selected_method == 'Random Forest (Recommended)':
             predi = RF_model.predict(vectorized_news)

            with st.spinner('Analysing news artical...'):
                time.sleep(random.uniform(1,2))
            with st.spinner('Thinking...'):
                time.sleep(random.uniform(1,2))
            with st.spinner('Predicting...'):
                time.sleep(random.uniform(1,2))

           # Display the predictions with increased font size for the last result
            st.write(f'<span style="font-size:30px;">{selected_method.replace(" (Recommended)", "")} Prediction: {"Fake News" if predi[0] == 0 else "True News"}</span>', unsafe_allow_html=True)


        else:
            st.warning('Please enter a news article.')
