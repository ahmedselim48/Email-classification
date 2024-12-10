from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pipeline
pipeline = joblib.load("text_clf_MIP_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    email_text = request.form['email_text']
    prediction = pipeline.predict([email_text])[0]
    result = "spam" if prediction == 1 else "ham"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
