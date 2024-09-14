import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from preprocess import load_data

def main():
    # Load the data
    data = load_data('data.txt')
    
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['message'])
    y = data['label'].apply(lambda x: 1 if x == 'spam' else 0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # Save the model and vectorizer
    joblib.dump(model, 'spam_classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    main()
