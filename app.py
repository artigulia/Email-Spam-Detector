import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower() # lower case
    text = nltk.word_tokenize(text) # tokenization
    y=[]
    for i in text: # Removing special characters and keeping only alphanumeric values
        if i.isalnum():
            y.append(i)
    #Removing stopwords
    text = y[:] # clong the y list into text
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    # Stemming
    text = y[:] # clong the y list into text again
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf=pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # Step 1. Pre-process
    transformed_sms = transform_text(input_sms)

    # Step 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # step 3. predict
    result = model.predict(vector_input)[0]

    # step 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")

