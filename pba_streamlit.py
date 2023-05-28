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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

    # Memisahkan data menjadi data pelatihan dan data pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Melatih model Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Memprediksi label untuk data pengujian
    y_pred = model.predict(X_test)

    # Menghitung akurasi model
    accuracy = accuracy_score(y_test, y_pred)

    # Menyimpan model ke dalam file pickle
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Menyimpan vectorizer ke dalam file pickle
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    st.success('Model berhasil dilatih dan disimpan ke dalam file sentiment_model.pkl')
    st.write('Akurasi Model:', accuracy)

# Fungsi untuk menganalisis sentimen menggunakan model Naive Bayes
def analyze_sentiment(text, model, vectorizer):
    # Preprocessing teks
    preprocessed_text = preprocess_text(text)

    # Melakukan vektorisasi teks menggunakan TF-IDF
    text_vectorized = vectorizer.transform([preprocessed_text])

    # Memprediksi sentimen
    sentiment = model.predict(text_vectorized)[0]

    # Mendapatkan polaritas menggunakan TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    return sentiment, polarity

# Fungsi utama Streamlit
def main():
    st.title('Analisis Sentimen Data Waralaba dari Tweet')

    # Tombol untuk melatih model
    if st.button('Latih Model'):
        train_model()

    # Memuat model dari file pickle
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Memuat vectorizer dari file pickle
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Kolom input teks untuk analisis sentimen
    st.subheader('Analisis Sentimen')
    review_text = st.text_input('Masukkan tweet tentang waralaba')

    if review_text and model and vectorizer:
        # Tombol untuk menganalisis sentimen
        if st.button('Analisis'):
            sentiment, polarity = analyze_sentiment(review_text, model, vectorizer)
            polarity_label = classify_polarity(polarity)
            st.write('Sentimen:', sentiment)
            st.write('Polarity:', polarity_label)

if __name__ == '__main__':
    main()
