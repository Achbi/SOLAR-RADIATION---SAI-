from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import re
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Global variables for model and scaler
model = None
scaler = None

def create_model():
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=15))
    model.add(Dropout(0.33))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.33))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.33))
    model.add(Dense(1, activation="relu"))
    model.compile(metrics=['mse'], loss='mae', optimizer=Adam(learning_rate=0.001))
    return model

def initialize():
    global model, scaler
    
    # Load your data
    data = pd.read_csv('SolarPrediction.csv.zip')
    
    # Preprocess the data
    df = preprocess_data(data)
    
    # Prepare features and target
    features = df.drop('Radiation', axis=1)
    target = df['Radiation']
    
    print("Training features:", features.columns)
    print("Number of features:", len(features.columns))
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create and train the model
    model = create_model()
    model.fit(features_scaled, target, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

def preprocess_data(data):
    df = data.copy()
    df['Month'] = pd.to_datetime(df['Data']).dt.month
    df['Day'] = pd.to_datetime(df['Data']).dt.day
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour
    df['Minute'] = pd.to_datetime(df['Time']).dt.minute
    df['Second'] = pd.to_datetime(df['Time']).dt.second
    df['risehour'] = df['TimeSunRise'].apply(lambda x: int(x.split(':')[0]))
    df['riseminute'] = df['TimeSunRise'].apply(lambda x: int(x.split(':')[1]))
    df['sethour'] = df['TimeSunSet'].apply(lambda x: int(x.split(':')[0]))
    df['setminute'] = df['TimeSunSet'].apply(lambda x: int(x.split(':')[1]))
    
    columns_to_drop = ['Data', 'Time', 'TimeSunRise', 'TimeSunSet']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])
    
    # Preprocess the input data
    input_features = preprocess_input(input_df)
    
    print("Input features:", input_features.columns)
    print("Number of input features:", len(input_features.columns))
    
    # Scale the features
    input_features_scaled = scaler.transform(input_features)
    
    # Make prediction
    prediction = model.predict(input_features_scaled)
    
    return jsonify({'prediction': float(prediction[0][0])})

def preprocess_input(df):
    df['UNIXTime'] = int(time.time())
    df['Month'] = pd.to_datetime(df['Data']).dt.month
    df['Day'] = pd.to_datetime(df['Data']).dt.day
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour
    df['Minute'] = pd.to_datetime(df['Time']).dt.minute
    df['Second'] = pd.to_datetime(df['Time']).dt.second
    df['risehour'] = df['TimeSunRise'].apply(lambda x: int(x.split(':')[0]))
    df['riseminute'] = df['TimeSunRise'].apply(lambda x: int(x.split(':')[1]))
    df['sethour'] = df['TimeSunSet'].apply(lambda x: int(x.split(':')[0]))
    df['setminute'] = df['TimeSunSet'].apply(lambda x: int(x.split(':')[1]))
    
    columns_to_drop = ['Data', 'Time', 'TimeSunRise', 'TimeSunSet']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    # Ensure the order of columns matches the training data
    expected_columns = ['UNIXTime', 'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'risehour', 'riseminute', 'sethour', 'setminute']
    df = df[expected_columns]
    
    return df

if __name__ == '__main__':
    initialize()
    app.run(debug=True)