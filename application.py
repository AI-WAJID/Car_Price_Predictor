from flask import Flask, request, jsonify, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and data
model = pickle.load(open('model/LR_model.pkl', 'rb'))
car_data = pd.read_csv('data/cleaned_car_data.csv')

@app.route('/')
def index():
    companies = sorted(car_data['company'].unique())
    years = sorted(car_data['year'].unique(), reverse=True)
    fuel_types = car_data['fuel_type'].unique()

    companies.insert(0, 'Select Company')

    return render_template('index.html',
                           companies=companies,
                           years=years,
                           fuel_types=fuel_types)

@app.route('/get_models', methods=['POST'])
@cross_origin()
def get_models():
    selected_company = request.form.get('company')
    models = sorted(car_data[car_data['company'] == selected_company]['name'].unique())
    return jsonify({'models': models})

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    prediction = model.predict(pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
    ))

    predicted_price = float(np.round(prediction[0], 2))

    return jsonify({'price': predicted_price})

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
