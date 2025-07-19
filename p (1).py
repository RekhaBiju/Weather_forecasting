import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle

# Train and save the model as a single file
def train_and_save_single_model(data_path, output_file="weather_model.pkl"):
    """
    Train the weather forecasting model and save it as a single file.
    
    Args:
        data_path (str): Path to the Excel file with weather data
        output_file (str): Filename to save the model to
    """
    print("Loading data...")
    # Load and preprocess data
    df = pd.read_excel(data_path)
    df = df.drop(columns=['snow', 'wpgt', 'tsun', 'prcp', 'wdir'], errors='ignore')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    print("Creating lag features...")
    # Create lag features
    for lag in range(1, 6):
        for col in ['tmax', 'tmin', 'tavg', 'pres', 'wspd']:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Define features and targets
    features = [col for col in df.columns if col not in ['date', 'tmax', 'tmin', 'tavg', 'pres', 'wspd']]
    targets = ['tmax', 'tmin', 'tavg', 'pres', 'wspd']
    
    X = df[features]
    y = df[targets]
    
    print("Splitting data into train/test sets...")
    # Split data
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    
    print("Training models (this may take a few minutes)...")
    # Train models
    models = {}
    for target in targets:
        print(f"Training model for {target}...")
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_train, y_train[target])
        models[target] = model
    
    # Create a single model package with everything needed for predictions
    model_package = {
        'models': models,
        'features': features,
        'targets': targets,
        'sample_data': df.tail(10)  # Keep a small sample for reference
    }
    
    print(f"Saving model to {output_file}...")
    # Save the complete model package to a single file
    with open(output_file, "wb") as f:
        pickle.dump(model_package, f)
    
    print(f"Model saved successfully to {output_file}")
    
    return output_file

# Main execution
if __name__ == "__main__":
    data_path = input("Enter the path to your weather data Excel file: ")
    output_file = "weather_model.pkl"
    
    # Train and save the model to a single file
    model_file = train_and_save_single_model(data_path, output_file)
    
    print("\nTo use the model, use this code:")
    print(f"""
import pickle
import pandas as pd
import numpy as np
import datetime

def predict_weather(input_date, model_file="{model_file}"):
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
    future_predictions = {{target: [] for target in targets}}

    for _ in range(5):
        # Prepare input features
        input_data = np.array([closest_date_row[features].values])
        
        # Predict for each target
        new_predictions = {{}}
        for target in targets:
            new_predictions[target] = models[target].predict(input_data)[0]
            future_predictions[target].append(new_predictions[target])

        # Update for next iteration
        for col in targets:
            closest_date_row[col] = new_predictions[col]
            
        for lag in range(5, 1, -1):
            for col in targets:
                closest_date_row[f"{{col}}_lag{{lag}}"] = closest_date_row[f"{{col}}_lag{{lag-1}}"]
        
        for col in targets:
            closest_date_row[f"{{col}}_lag1"] = new_predictions[col]

    return pd.DataFrame({{'date': future_dates, **future_predictions}})

# Example usage:
date_input = "2023-10-15"  # Your date here
forecast = predict_weather(date_input)
print(forecast)
""")