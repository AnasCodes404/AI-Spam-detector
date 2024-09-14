import joblib
from preprocess import preprocess_message

def main():
    # Load the saved model and vectorizer
    model = joblib.load('spam_classifier.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    
    # Input message for prediction
    message = input("Enter a message to classify: ")
    
    # Preprocess the message
    processed_message = preprocess_message(message)
    
    # Transform the message using the saved vectorizer
    X = vectorizer.transform([processed_message])
    
    # Predict whether the message is spam or not
    prediction = model.predict(X)
    if prediction[0] == 1:
        print("The message is classified as: SPAM")
    else:
        print("The message is classified as: HAM")

if __name__ == "__main__":
    main()
