
import joblib
import os
import pandas as pd
import numpy as np

def model_fn(model_dir):
    """Load the model and preprocessing artifacts from the model_dir."""
    model = joblib.load(os.path.join(model_dir, "random_forest_model.joblib"))
    ordinal_encoders = joblib.load(os.path.join(model_dir, "ordinal_encoders.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return {
        'model': model,
        'ordinal_encoders': ordinal_encoders,
        'scaler': scaler
    }

def input_fn(request_body, request_content_type):
    """Parse the incoming request."""
    if request_content_type == 'application/json':
        data = pd.read_json(request_body)
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_and_artifacts):
    """Preprocess the input data and make predictions."""
    model = model_and_artifacts['model']
    ordinal_encoders = model_and_artifacts['ordinal_encoders']
    scaler = model_and_artifacts['scaler']
    
    # Encode categorical variables
    categorical_columns = ['zipcode', 'city', 'state', 'county']
    for col in categorical_columns:
        if col in input_data.columns:
            input_data[col] = ordinal_encoders[col].transform(input_data[[col]])
        else:
            raise ValueError(f"Missing expected column: {col}")
    
    # Scale numerical features
    numerical_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'price_per_sqft', 
                         'avg_price_by_city', 'avg_price_per_sqft_by_city']
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
    
    # Make predictions
    predictions = model.predict(input_data)
    return predictions
