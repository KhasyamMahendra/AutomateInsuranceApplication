from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('insurance_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from HTML form
        age = int(request.form['age'])
        bmi = float(request.form['bmii'])
        smoker_status = request.form['smoker']

        # Convert smoker text to number (Yes=1, No=0)
        smoker = 1 if smoker_status == 'yes' else 0

        # Make prediction
        input_data = np.array([[age, bmi, smoker]])
        prediction = model.predict(input_data)[0]

        # Format the result nicely
        result_text = f"${prediction:,.2f}"

        return render_template('index.html', prediction=result_text, age=age, bmi=bmi, smoker=smoker_status)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
