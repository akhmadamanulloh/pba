import pandas as pd
import re
import string
import pickle
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk melakukan preprosesing teks
def preprocess_text(text):
    # Case folding
    text = text.lower()
    
    # Filtering
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    # Tokenizing
    tokens = word_tokenize(text)
    
    # Menghapus stop words
    stop_words = set(stopwords.words("english")) | set(stopwords.words("indonesian"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Menggabungkan kembali kata-kata yang telah diproses
    preprocessed_text = ' '.join(filtered_tokens)
    
    return preprocessed_text

# Membaca data CSV dan menyimpan ke dalam DataFrame
df = pd.read_csv('data_tweet.csv')

# Melakukan analisis sentimen pada setiap tweet
def get_sentiment(tweet):
    # Menghitung polaritas tweet
    polarity = TextBlob(tweet).sentiment.polarity
    
    # Menentukan sentimen tweet berdasarkan nilai polaritas
    if polarity > 0:
        return '1'
    elif polarity < 0:
        return '-1'
    else:
        return '0'

df['sentiment'] = df['Tweet'].apply(get_sentiment)

# Melakukan preprosesing pada teks ulasan
df['preprocessed_text'] = df['Tweet'].apply(preprocess_text)

# Melakukan analisis sentimen menggunakan Naive Bayes
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['preprocessed_text'])
y = df['sentiment']

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X, y)

# Melakukan analisis sentimen menggunakan Neural Network
neural_network_model = MLPClassifier()
neural_network_model.fit(X, y)

# Simpan model Naive Bayes dalam file pickle
with open('naive_bayes_model.pickle', 'wb') as file:
    pickle.dump(naive_bayes_model, file)

# Simpan model Neural Network dalam file pickle
with open('neural_network_model.pickle', 'wb') as file:
    pickle.dump(neural_network_model, file)

# Implementasi aplikasi Streamlit
st.title("Analisis Sentimen Waralaba")
text_input = st.text_input("Masukkan teks ulasan:")
model_choice = st.selectbox("Pilih Model", ["Naive Bayes", "Neural Network"])

if st.button("Delete Pickle"):
    try:
        # Hapus file pickle
        os.remove('naive_bayes_model.pickle')
        os.remove('neural_network_model.pickle')
        st.write("Pickle file deleted.")
    except FileNotFoundError:
        st.write("Pickle file not found.")

if st.button("Create Model"):
    if model_choice == "Naive Bayes":
        # Buat model Naive Bayes
        with open('naive_bayes_model.pickle', 'wb') as file:
            pickle.dump(naive_bayes_model, file)
        st.write("Naive Bayes model created.")
    elif model_choice == "Neural Network":
        # Buat model Neural Network
        with open('neural_network_model.pickle', 'wb') as file:
            pickle.dump(neural_network_model, file)
        st.write("Neural Network model created.")

if st.button("Prediksi Sentimen"):
    if text_input:
        preprocessed_text = preprocess_text(text_input)
        text_vectorized = vectorizer.transform([preprocessed_text])
        
        if model_choice == "Naive Bayes":
            with open('naive_bayes_model.pickle', 'rb') as file:
                loaded_model = pickle.load(file)
            sentiment = loaded_model.predict(text_vectorized)[0]
        elif model_choice == "Neural Network":
            with open('neural_network_model.pickle', 'rb') as file:
                loaded_model = pickle.load(file)
            sentiment = loaded_model.predict(text_vectorized)[0]
        
        st.write("Sentimen: ", sentiment)
    else:
        st.write("Masukkan teks ulasan untuk melakukan prediksi sentimen.")
