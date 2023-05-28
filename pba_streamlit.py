import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk melakukan preprocessing teks
def preprocess_text(text):
    

# Tentukan jalur lengkap ke file pkl yang ingin dihapus
file_path = 'sentiment_model.pkl'

# Periksa apakah file ada sebelum menghapusnya
if os.path.exists(file_path):
    os.remove(file_path)
    print("File berhasil dihapus.")
else:
    print("File tidak ditemukan.")
    # Case folding
    text = text.lower()

    # Filtering dan tokenizing
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

# Fungsi untuk mendapatkan sentimen menggunakan TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positif'
    elif sentiment < 0:
        return 'Negatif'
    else:
        return 'Netral'

# Fungsi untuk melatih model Naive Bayes dan menyimpannya ke dalam file pickle
def train_model():
    # Baca data CSV
    df = pd.read_csv('data_tweet.csv')

    # Preprocessing teks
    df['Preprocessed_Text'] = df['Tweet'].apply(preprocess_text)

    # Mendapatkan sentimen
    df['sentiment'] = df['Tweet'].apply(get_sentiment)

    # Memisahkan fitur dan label
    X = df['Preprocessed_Text']
    y = df['sentiment']

    # Melakukan vektorisasi teks menggunakan TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    # Melatih model Naive Bayes
    model = MultinomialNB()
    model.fit(X, y)
    
    # Menyimpan model ke dalam file pickle
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Menyimpan vectorizer ke dalam file pickle
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    st.success('Model berhasil dilatih dan disimpan ke dalam file sentiment_model.pkl')

# Fungsi utama Streamlit
def main():
    st.title('Analisis Sentimen Data Waralaba dari Tweet')

    # Tombol untuk melatih model
    if st.button('Latih Model'):
        train_model()

    # Memuat model dari file pickle
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Memuat vectorizer dari file pickle
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        model = None
        vectorizer = None

    # Kolom input teks untuk analisis sentimen
    st.subheader('Analisis Sentimen')
    review_text = st.text_input('Masukkan tweet tentang waralaba')

    # Tombol untuk menganalisis sentimen
    if st.button('Analisis', disabled=model is None or vectorizer is None):
        if model is not None and vectorizer is not None:
            sentiment = get_sentiment(review_text)
            st.write('Sentimen:', sentiment)
        else:
            st.error('Model belum dilatih. Silakan klik tombol "Latih Model" terlebih dahulu.')

if __name__ == '__main__':
    main()
