import pandas as pd
import re
import string
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os

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
        return 'positif'
    elif polarity < 0:
        return 'negatif'
    else:
        return 'netral'

df['sentiment'] = df['Tweet'].apply(get_sentiment)

# Melakukan preprosesing pada teks ulasan
df['preprocessed_text'] = df['Tweet'].apply(preprocess_text)

# Melakukan analisis sentimen menggunakan Naive Bayes
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['preprocessed_text'])
y = df['sentiment']

# Memisahkan data menjadi data latih (train) dan data uji (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Naive Bayes
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

# Membangun model Neural Network
neural_network_model = MLPClassifier()
neural_network_model.fit(X_train, y_train)

# Simpan model Naive Bayes dalam file pickle
with open('naive_bayes_model.pickle', 'wb') as file:
    pickle.dump(naive_bayes_model, file)

# Simpan model Neural Network dalam file pickle
with open('neural_network_model.pickle', 'wb') as file:
    pickle.dump(neural_network_model, file)

# Implementasi aplikasi Streamlit
st.title("Analisis Sentimen Waralaba")
text_input = st.text_input("Masukkan teks ulasan:")
model_choice = st.radio("Pilih Model", ('Naive Bayes', 'Neural Network'))

if st.button("Delete Pickle"):
    if model_choice == 'Naive Bayes':
        os.remove('naive_bayes_model.pickle')
        st.write("Naive Bayes model deleted.")
    elif model_choice == 'Neural Network':
        os.remove('neural_network_model.pickle')
        st.write("Neural Network model deleted.")

if model_choice == 'Naive Bayes' and os.path.isfile('naive_bayes_model.pickle'):
    loaded_model = pickle.load(open('naive_bayes_model.pickle', 'rb'))
elif model_choice == 'Neural Network' and os.path.isfile('neural_network_model.pickle'):
    loaded_model = pickle.load(open('neural_network_model.pickle', 'rb'))

if st.button("Prediksi Sentimen"):
    if text_input:
        preprocessed_text = preprocess_text(text_input)
        text_vectorized = vectorizer.transform([preprocessed_text])
        
        if model_choice == 'Naive Bayes' and os.path.isfile('naive_bayes_model.pickle'):
            sentiment = loaded_model.predict(text_vectorized)[0]
        elif model_choice == 'Neural Network' and os.path.isfile('neural_network_model.pickle'):
            sentiment = loaded_model.predict(text_vectorized)[0]
        else:
            st.write("Model not found. Create the model first.")
            sentiment = None
        
        if sentiment is not None:
            st.write("Sentimen: ", sentiment)
    else:
        st.write("Masukkan teks ulasan untuk melakukan prediksi sentimen.")
