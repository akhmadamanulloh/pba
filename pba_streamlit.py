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

model = MultinomialNB()
model.fit(X, y)

# Simpan model dalam file pickle
with open('sentiment_model.pickle', 'wb') as file:
    pickle.dump(model, file)

# Implementasi aplikasi Streamlit
st.title("Analisis Sentimen Waralaba")
text_input = st.text_input("Masukkan teks ulasan:")
if st.button("Prediksi Sentimen"):
    if text_input:
        preprocessed_text = preprocess_text(text_input)
        text_vectorized = vectorizer.transform([preprocessed_text])
        sentiment = model.predict(text_vectorized)[0]
        st.write("Sentimen: ", sentiment)
    else:
        st.write("Masukkan teks ulasan untuk melakukan prediksi sentimen.")

if st.button("Memanggil File Pickle"):
    with open('sentiment_model.pickle', 'rb') as file:
        loaded_model = pickle.load(file)
    st.write("Model Sentimen dari File Pickle: ", loaded_model)
