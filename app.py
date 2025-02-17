import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

st.title("Fake News Detector")
st.write("Enter a News Article to Analyze")

news_input = st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        
        if prediction[0]==0:
            st.success("Real News")
        else:
            st.error("Fake News")
            
    else:
        st.warning("Please Enter News Article to Analyze!")