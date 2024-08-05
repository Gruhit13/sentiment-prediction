import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
import keras

vectorizer = keras.layers.TextVectorization(
    max_tokens = 2000,
    output_sequence_length = 32
)
vectorizer.load_assets('./vectorizer')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Get english stopwords
en_stopwords = set(stopwords.words('english'))

# Get the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Conver the text to lowercase
    text = text.lower()

    # Replace '#' tags
    text = text.replace('#', '')

    # Remove the nametags/mentions
    text = re.sub(r'@[^\s]+', '', text)

    # Remove the hyperlinks
    text = re.sub(r'https:\/\/\S+', '', text)

    # Remove the leading and trailing spaces
    text = text.strip()

    # Remove the emojis
    text = emoji.demojize(text)

    # Tokenize the word to lematize it
    tokens = nltk.word_tokenize(text)
    lemma_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemma_tokens = [w for w in lemma_tokens if w not in en_stopwords]

    text = ' '.join(lemma_tokens)

    tokens = vectorizer(text)
    tokens = tf.expand_dims(tokens, axis=0)

    return tokens

if __name__ == "__main__":
    print(preprocess_text("I am running today"))