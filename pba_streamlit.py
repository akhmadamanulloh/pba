import pandas as pd
import re
import string
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk melakukan preprocessing teks
def preprocess_text(text):
    # Case folding
    text = text.lower()

    # Filtering
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenizing
    tokens = word_tokenize(text)

    # Menghapus stop words
    additional_stopwords = ['gtu', 'yg', 'adlh', 'yaa', 'adh', 'akn']
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Menggabungkan kembali kata-kata yang telah diproses
    preprocessed_text = ' '.join(stemmed_tokens)

    return preprocessed_text

# Mengambil daftar stop words
stop_words = set(stopwords.words("english")) | set(stopwords.words("indonesian"))

# Inisialisasi objek PorterStemmer
stemmer = PorterStemmer()

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

# Melakukan preprocessing pada teks ulasan
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

if model_choice == 'Naive Bayes':
    loaded_model = pickle.load(open('naive_bayes_model.pickle', 'rb'))
elif model_choice == 'Neural Network':
    loaded_model = pickle.load(open('neural_network_model.pickle', 'rb'))

if st.button("Prediksi Sentimen"):
    if text_input:
        preprocessed_text = preprocess_text(text_input)
        text_vectorized = vectorizer.transform([preprocessed_text])
        sentiment = loaded_model.predict(text_vectorized)[0]
        st.write("Sentimen: ", sentiment)

        # Menampilkan hasil preprocessing
        st.subheader("Hasil Preprocessing:")
        st.write("Case Folding: ", preprocessed_text.lower())
        st.write("Filtering: ", re.sub(r'[^a-zA-Z\s]', '', preprocessed_text))
        st.write("Tokenizing: ", word_tokenize(preprocessed_text))
        st.write("Stop Words: ", [token for token in word_tokenize(preprocessed_text) if token in stop_words])
        st.write("Stemming: ", [stemmer.stem(token) for token in word_tokenize(preprocessed_text)])

        # Menghitung akurasi model
        y_pred = loaded_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Akurasi Model:")
        st.write("Akurasi: ", accuracy)
    else:
        st.write("Masukkan teks ulasan untuk melakukan prediksi sentimen.")
