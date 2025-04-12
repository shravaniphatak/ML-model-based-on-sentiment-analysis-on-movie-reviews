from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
import os

print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir())

classifier = pickle.load(open('Movies_Review_classification.pkl', 'rb'))
cv = pickle.load(open('Tfidf-Vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            message = request.form['message']
            print("Received message:", message)  # Add this
            data = [message]
            vect = cv.transform(data).toarray()
            prediction = classifier.predict(vect)
            sentiment = "Positive Review 😊" if prediction[0] == 1 else "Negative Review 😞"
            return render_template('home.html', prediction=sentiment, user_input=message)
    except Exception as e:
        print("Error occurred:", e)
        return "Something went wrong: " + str(e)


if __name__ == '__main__':
    app.run(debug=True)
