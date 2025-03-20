from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime

# Initialize Flask App
app = Flask(__name__)

# Load trained XGBoost model
model = joblib.load("xgboost_accident_model.pkl")

# Load historical accident dataset
df = pd.read_csv("US_Accidents_MA.csv")

# Extract required location and infrastructure features
df_infra = df[["Start_Lat", "Start_Lng", "Traffic_Signal", "Junction", "Crossing", "Amenity",
               "Bump", "Give_Way", "No_Exit", "Railway", "Station", "Stop", "Traffic_Calming"]].drop_duplicates().fillna(0)

# Get current month & week for prediction
current_month = datetime.now().month
current_week = datetime.now().isocalendar()[1]

# Default duration value (in seconds)
DEFAULT_DURATION = 300  # 5 minutes

# Weather condition mapping (One-Hot Encoding)
weather_map = ["Weather_Fair", "Weather_Cloudy", "Weather_Clear", "Weather_Overcast",
               "Weather_Snow", "Weather_Haze", "Weather_Rain", "Weather_Thunderstorm",
               "Weather_Windy", "Weather_Hail", "Weather_Thunder"]

# Wind direction mapping (One-Hot Encoding)
wind_map = ["Wind_C", "Wind_E", "Wind_N", "Wind_S", "Wind_V", "Wind_W"]

# Define feature order (must match training)
feature_columns = [
    'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
    'Visibility(mi)', 'Wind_Speed(mph)', 'Amenity', 'Bump', 'Crossing',
    'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Station', 'Stop',
    'Traffic_Calming', 'Traffic_Signal', 'Duration', 'Month', 'Week', 'Hour',
    'Weather_Fair', 'Weather_Cloudy', 'Weather_Clear', 'Weather_Overcast',
    'Weather_Snow', 'Weather_Haze', 'Weather_Rain', 'Weather_Thunderstorm',
    'Weather_Windy', 'Weather_Hail', 'Weather_Thunder', 'Wind_C', 'Wind_E',
    'Wind_N', 'Wind_S', 'Wind_V', 'Wind_W'
]

# Configure logging
logging.basicConfig(filename="flask.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from form
        user_input = {
            "Hour": int(request.form["Hour"]),
            "Month": int(request.form.get("Month", current_month)),
            "Week": int(request.form.get("Week", current_week)),
            "Weather_Condition": request.form["Weather_Condition"],
            "Temperature(F)": float(request.form["Temperature"]),
            "Humidity(%)": float(request.form["Humidity"]),
            "Pressure(in)": float(request.form["Pressure"]),
            "Visibility(mi)": float(request.form["Visibility"]),
            "Wind_Condition": request.form["Wind_Condition"],
            "Wind_Speed(mph)": float(request.form["Wind_Speed"])
        }

        # Prepare prediction datasets for each severity
        severity_data = {1: [], 2: [], 3: [], 4: []}

        for _, row in df_infra.iterrows():
            # Create infrastructure feature array
            input_features = list(row[f] for f in df_infra.columns)

            # Append user inputs
            input_features += [
                user_input["Temperature(F)"], user_input["Humidity(%)"],
                user_input["Pressure(in)"], user_input["Visibility(mi)"],
                user_input["Wind_Speed(mph)"], DEFAULT_DURATION,
                user_input["Month"], user_input["Week"], user_input["Hour"]
            ]

            # One-Hot Encoding for Weather Condition
            weather_one_hot = [1 if user_input["Weather_Condition"] == w else 0 for w in weather_map]
            input_features += weather_one_hot

            # One-Hot Encoding for Wind Condition
            wind_one_hot = [1 if user_input["Wind_Condition"] == w else 0 for w in wind_map]
            input_features += wind_one_hot

            # ✅ Convert NumPy array to Pandas DataFrame with proper feature names
            input_df = pd.DataFrame([input_features], columns=feature_columns)

            # ✅ Predict severity and apply reverse label shift (+1)
            severity_prediction = int(model.predict(input_df)[0]) 

            # Store predictions separately by severity
            if severity_prediction in severity_data:
                severity_data[severity_prediction].append({
                    "latitude": row["Start_Lat"],
                    "longitude": row["Start_Lng"]
                })

        return render_template('predicted_map.html', accident_data=json.dumps(severity_data), user_inputs=user_input)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return f"Error: {str(e)}"


@app.route('/map')
def show_map():
    return render_template('predicted_map.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
