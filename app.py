from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    message = data.get('message', '')

    # Process the message
    X = vectorizer.transform([message])
    prediction = model.predict(X)

    # Return the result
    result = 'spam' if prediction[0] == 1 else 'not spam'
    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True)
