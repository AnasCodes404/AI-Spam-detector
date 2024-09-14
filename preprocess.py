import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def preprocess_message(message):
    stemmer = PorterStemmer()
    message = message.lower()
    message = ''.join([char for char in message if char.isalnum() or char.isspace()])
    words = message.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'message'])
    data['message'] = data['message'].apply(preprocess_message)
    return data
