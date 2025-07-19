import pickle
import pandas as pd
import numpy as np
import datetime

def predict_weather(input_date, model_file="weather_model.pkl"):
    # Load the model package
    with open(model_file, "rb") as f:
        model_package = pickle.load(f)
    
    # Extract components
    models = model_package['models']
    features = model_package['features']
    targets = model_package['targets']
    sample_df = model_package['sample_data']
    
    # Convert input date
    input_date = pd.to_datetime(input_date)
    
    # Prepare data row for prediction
    closest_date_row = sample_df.iloc[-1].copy()
    closest_date_row['date'] = input_date
    
    # Forecast next 5 days
    future_dates = [closest_date_row['date'] + datetime.timedelta(days=i) for i in range(1, 6)]
    future_predictions = {target: [] for target in targets}

    for _ in range(5):
        # Prepare input features
        input_data = np.array([closest_date_row[features].values])
        
        # Predict for each target
        new_predictions = {}
        for target in targets:
            new_predictions[target] = models[target].predict(input_data)[0]
            future_predictions[target].append(new_predictions[target])

        # Update for next iteration
        for col in targets:
            closest_date_row[col] = new_predictions[col]
            
        for lag in range(5, 1, -1):
            for col in targets:
                closest_date_row[f"{col}_lag{lag}"] = closest_date_row[f"{col}_lag{lag-1}"]
        
        for col in targets:
            closest_date_row[f"{col}_lag1"] = new_predictions[col]

    return pd.DataFrame({'date': future_dates, **future_predictions})

# Example usage:
date_input = "2023-10-15"  # Your date here
forecast = predict_weather(date_input)
print(forecast)