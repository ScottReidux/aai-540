
import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def model_fn(model_dir):
    """Load the model for inference"""
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    encoders = joblib.load(os.path.join(model_dir, 'encoders.joblib'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    return {'model': model, 'encoders': encoders, 'scaler': scaler}

def input_fn(input_data, content_type):
    import pandas as pd
    import io
    if content_type == 'text/csv':
        df = pd.read_csv(io.StringIO(input_data), header=None)
        # Assign column names
        df.columns = ['bedrooms', 'bathrooms', 'sqft_living', 'price_per_sqft',
                      'zipcode', 'city', 'state', 'county', 'avg_price_by_city',
                      'avg_price_per_sqft_by_city']
        return df
    else:
        raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(input_data, model_objects):
    model = model_objects['model']
    encoders = model_objects['encoders']
    scaler = model_objects['scaler']
    
    # Preprocess input data
    for col in ['zipcode', 'city', 'state', 'county']:
        # Map categories to integers; unknown categories get -1
        input_data[col] = input_data[col].map(encoders[col]).fillna(-1)
    
    numerical_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'price_per_sqft',
                         'avg_price_by_city', 'avg_price_per_sqft_by_city']
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
    
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, accept):
    if accept == 'text/csv':
        return '\n'.join(map(str, prediction))
    else:
        raise ValueError(f'Unsupported accept type: {accept}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', './data'))
    parser.add_argument('--n-estimators', type=int, default=100)

    args = parser.parse_args()

    # Read the training data
    train_data_files = [os.path.join(args.train, file) for file in os.listdir(args.train) if file.endswith('.csv')]
    df_train = pd.concat([pd.read_csv(file) for file in train_data_files])

    # Read the validation data
    validation_data_files = [os.path.join(args.validation, file) for file in os.listdir(args.validation) if file.endswith('.csv')]
    df_validation = pd.concat([pd.read_csv(file) for file in validation_data_files])

    # Prepare the data
    X_train = df_train.drop(columns=['id', 'event_time', 'price'])
    y_train = df_train['price']
    X_validation = df_validation.drop(columns=['id', 'event_time', 'price'])
    y_validation = df_validation['price']

    # Handle categorical variables using pd.factorize
    encoders = {}
    for col in ['zipcode', 'city', 'state', 'county']:
        # Fit factorizer on training data
        categories, uniques = pd.factorize(X_train[col])
        X_train[col] = categories
        encoders[col] = {k: v for v, k in enumerate(uniques)}
        # Map validation data using the same mapping; unknowns get -1
        X_validation[col] = X_validation[col].map(encoders[col]).fillna(-1)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'price_per_sqft',
                         'avg_price_by_city', 'avg_price_per_sqft_by_city']
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_validation[numerical_columns] = scaler.transform(X_validation[numerical_columns])

    # Train the model
    model = RandomForestRegressor(n_estimators=args.n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and preprocessing objects
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    joblib.dump(encoders, os.path.join(args.model_dir, 'encoders.joblib'))
    joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.joblib'))
