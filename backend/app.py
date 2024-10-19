import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Allowing CORS for all routes, you can restrict it to a specific origin like "http://localhost:3000"
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables to store our models and scaler
xgb_model = None
nn_model = None
scaler = None

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    df = data.copy()
    
    # Date and time processing
    df['Data'] = pd.to_datetime(df['Data'])
    df['Month'] = df['Data'].dt.month
    df['Day'] = df['Data'].dt.day
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour
    df['Minute'] = pd.to_datetime(df['Time']).dt.minute
    df['Second'] = pd.to_datetime(df['Time']).dt.second
    
    # Sun rise and set time processing
    df['risehour'] = df['TimeSunRise'].apply(lambda x: int(x.split(':')[0]))
    df['riseminute'] = df['TimeSunRise'].apply(lambda x: int(x.split(':')[1]))
    df['sethour'] = df['TimeSunSet'].apply(lambda x: int(x.split(':')[0]))
    df['setminute'] = df['TimeSunSet'].apply(lambda x: int(x.split(':')[1]))
    
    # Drop unnecessary columns
    columns_to_drop = ['UNIXTime', 'Time', 'TimeSunRise', 'TimeSunSet']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    return df

def transform_features(df):
    transform = {
        'Temperature': np.log(df['Temperature'] + 1),
        'Pressure': stats.boxcox(df['Pressure'] + 1)[0],
        'Humidity': stats.boxcox(df['Humidity'] + 1)[0],
        'Speed': np.log(df['Speed'] + 1),
        'WindDirection': MinMaxScaler().fit_transform(df[['WindDirection(Degrees)']]).flatten()
    }
    
    for col, transformed_data in transform.items():
        df[col] = transformed_data
    
    return df

def train_models(X, y):
    global xgb_model, nn_model, scaler
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(learning_rate=0.1, max_depth=8)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Train Neural Network model
    nn_model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_dim=X_train.shape[1]),
        keras.layers.Dropout(0.33),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.33),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.33),
        keras.layers.Dense(1, activation="relu")
    ])
    nn_model.compile(loss='mae', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mse'])
    nn_model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate models
    xgb_pred = xgb_model.predict(X_test_scaled)
    nn_pred = nn_model.predict(X_test_scaled).flatten()
    
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
    nn_r2 = r2_score(y_test, nn_pred)
    
    print(f"XGBoost - RMSE: {xgb_rmse:.2f}, R2: {xgb_r2:.2f}")
    print(f"Neural Network - RMSE: {nn_rmse:.2f}, R2: {nn_r2:.2f}")
    
    # Save models and scaler
    joblib.dump(xgb_model, 'xgb_model.joblib')
    nn_model.save('nn_model.h5')
    joblib.dump(scaler, 'scaler.joblib')

@app.route('/train', methods=['POST'])
def train():
    try:
        # Load and preprocess data
        df = load_and_preprocess_data('SolarPrediction.csv')
        df = transform_features(df)
        
        # Prepare features and target
        X = df.drop('Radiation', axis=1)
        y = df['Radiation']
        
        # Train models
        train_models(X, y)
        
        return jsonify({'message': 'Models trained successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    global xgb_model, nn_model, scaler
    
    try:
        # Load models and scaler if not already loaded
        if xgb_model is None:
            xgb_model = joblib.load('xgb_model.joblib')
        if nn_model is None:
            nn_model = keras.models.load_model('nn_model.h5')
        if scaler is None:
            scaler = joblib.load('scaler.joblib')
        
        # Get input data from request
        input_data = request.json
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Transform features
        input_df = transform_features(input_df)
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make predictions using both models
        xgb_prediction = xgb_model.predict(input_scaled)[0]
        nn_prediction = nn_model.predict(input_scaled)[0][0]
        
        # Average the predictions
        final_prediction = (xgb_prediction + nn_prediction) / 2
        
        return jsonify({
            'prediction': float(final_prediction),
            'xgb_prediction': float(xgb_prediction),
            'nn_prediction': float(nn_prediction)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/historical_data', methods=['GET'])
def get_historical_data():
    try:
        df = load_and_preprocess_data('SolarPrediction.csv')
        sample = df.sample(100)  # Return a sample of 100 data points
        return jsonify(sample[['Data', 'Radiation']].to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
