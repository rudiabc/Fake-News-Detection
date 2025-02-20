import pandas as pd
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
        
        tf_idf = vectorizer.get_feature_names_out()
        df_input_tfidf = pd.DataFrame(transform_input.toarray(), columns=tf_idf)
        
        if prediction[0]==0:
            st.success("Real News")
        else:
            st.error("Fake News")
            
        st.subheader("Result of Vectorization using TF-IDF Vectorizer")
        st.dataframe(df_input_tfidf)
        
            
    else:
        st.warning("Please Enter News Article to Analyze!")