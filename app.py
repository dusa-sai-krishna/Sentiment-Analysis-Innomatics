import streamlit as st
from utilities import text_preprocessing,text_vectorizer,getSentiment

st.title("Sentiment Predictor using Word Embeddings")
corpus=st.text_area("Enter text here for Sentiment Analysis")

st.header('Entered Text')
st.write(corpus)

cleaned_text=text_preprocessing(corpus,flag="lemmatization")

st.header('Cleaned Text')
st.write(cleaned_text)

#Vectorization
vectorized_text=text_vectorizer(cleaned_text)


# Prediction of Sentiment
sentiment=getSentiment(vectorized_text) 
st.header('Sentiment')
if sentiment==1:
    st.write("Positive")
else:
    st.write("Negative")
