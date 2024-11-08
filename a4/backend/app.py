from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from custom_perceptron import CustomPerceptron

app = Flask(__name__)

cors=CORS(app)

df = pd.read_csv('./diabetes.csv')
X = df.drop(columns=['Outcome'])
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


with open('naive_bayes_model.pkl', 'rb') as file:
    naive_bayes_model = pickle.load(file)

with open('perceptron_model.pkl', 'rb') as file:
    perceptron_model = pickle.load(file)

with open('custom_perceptron_model.pkl', 'rb') as file:
    custom_perceptron_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_nb', methods=['POST'])
def predict_nb():
    data = request.get_json()
    features = [
        data['Glucose'],
        data['Insulin'],
        data['BMI'],
        data['Age']
    ]
    features== np.array(features).reshape(1, -1)
    features_scaled = scaler.transform([features])

    prediction = naive_bayes_model.predict(features_scaled)

    return jsonify({'model':'naive_bayes','prediction': int(prediction[0])})

@app.route('/predict_perceptron', methods=['POST'])
def predict_perceptron():
    data = request.get_json()
    features = [
        data['Glucose'],
        data['Insulin'],
        data['BMI'],
        data['Age']
    ]
    features== np.array(features).reshape(1, -1)
    features_scaled = scaler.transform([features])

    prediction = perceptron_model.predict(features_scaled)

    return jsonify({'model':'perceptron','prediction': int(prediction[0])})

@app.route('/predict_csm', methods=['POST'])
def perdict_csm():
    data = request.get_json()
    features = [
        data['Glucose'],
        data['Insulin'],
        data['BMI'],
        data['Age']
    ]
    features== np.array(features).reshape(1, -1)
    features_scaled = scaler.transform([features])

    prediction = custom_perceptron_model.predict(features_scaled)

    return jsonify({'model':'custom_perceptron','prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
