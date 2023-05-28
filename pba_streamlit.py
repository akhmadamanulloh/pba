import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk melakukan preprocessing teks
def preprocess_text(text):
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

# Fungsi untuk menghapus file pkl sentimen sebelumnya
def delete_sentiment_model():
    file_path = 'sentiment_model.pkl'
    if os.path.exists(file_path):
        os.remove(file_path)
        st.success("File sentiment_model.pkl berhasil dihapus.")
    else:
        st.warning("File sentiment_model.pkl tidak ditemukan.")

# Fungsi untuk melatih model Neural Network dan menyimpannya ke dalam file pickle
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

    # Melatih model Neural Network
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    y_one_hot = pd.get_dummies(y)
    model.fit(X, y_one_hot, epochs=10, batch_size=32)

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

            # Melakukan prediksi menggunakan model Neural Network
            y_pred = model.predict(X)
            predicted_sentiment = y_pred.argmax(axis=1)
            sentiment_mapping = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
            predicted_sentiment = sentiment_mapping[predicted_sentiment[0]]
            
            st.write('Sentimen Prediksi:', predicted_sentiment)
        else:
            st.error('Model belum dilatih atau belum dimuat. Silakan klik tombol "Latih Model" atau "Import Model" terlebih dahulu.')

if __name__ == '__main__':
    main()
