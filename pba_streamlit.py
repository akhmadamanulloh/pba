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

# Download stopword Bahasa Indonesia
nltk.download('stopwords')
stop_words_id = set(stopwords.words('indonesian'))

# Download stopword Bahasa Inggris
stop_words_en = set(stopwords.words('english'))

# Fungsi untuk melakukan preprocessing teks
def preprocess_text(text):
    # Case folding
    text = text.lower()

    # Filtering dan tokenizing
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words_id and token not in stop_words_en]

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

# Fungsi untuk menghapus file pkl sentimen sebelumnya
def delete_sentiment_model():
    file_path = 'sentiment_model.pkl'
    if os.path.exists(file_path):
        os.remove(file_path)
        st.success("File sentiment_model.pkl berhasil dihapus.")
    else:
        st.warning("File sentiment_model.pkl tidak ditemukan.")

# Fungsi untuk melatih model Naive Bayes dan menyimpannya ke dalam file pickle
def train_model():
    # Baca data CSV
    df = pd.read_csv('data_tweet.csv')

    # Preprocessing teks
    df['Preprocessed_Text'] = df['Tweet'].apply(preprocess_text)

    # Memisahkan fitur dan label
    X = df['Preprocessed_Text']
    y = df['Label']

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

# Fungsi untuk memuat model dari file pickle
def load_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Fungsi utama Streamlit
def main():
    st.title('Analisis Sentimen Data Waralaba dari Tweet')

    # Tombol untuk menghapus model
    if st.button('Hapus Model'):
        delete_sentiment_model()

    # Tombol untuk melatih model
    if st.button('Latih Model'):
        train_model()

    # Memuat model dari file pickle
    model, vectorizer = load_model()

    # Kolom input teks untuk analisis sentimen
    st.subheader('Analisis Sentimen')
    review_text = st.text_input('Masukkan tweet tentang waralaba')

    # Tombol untuk menganalisis sentimen
    if st.button('Analisis', disabled=model is None or vectorizer is None):
        if model is not None and vectorizer is not None:
            sentiment = get_sentiment(review_text)
            st.write('Sentimen Asli:', sentiment)

            preprocessed_text = preprocess_text(review_text)
            X = vectorizer.transform([preprocessed_text])
            predicted_sentiment = model.predict(X)[0]
            st.write('Sentimen Prediksi:', predicted_sentiment)
        else:
            st.error('Model belum dilatih atau belum dimuat. Silakan klik tombol "Latih Model" atau "Import Model" terlebih dahulu.')

if __name__ == '__main__':
    main()
