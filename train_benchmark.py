
# train_benchmark.py

import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

def model_fn(model_dir):
    """Load the model for inference"""
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

def input_fn(input_data, content_type):
    import pandas as pd
    import io
    import numpy as np

    if content_type == 'text/csv':
        df = pd.read_csv(io.StringIO(input_data), header=None)
        df.columns = ['sqft_living']
        return df
    elif content_type == 'application/x-npy':
        # Handle numpy array format
        data = np.load(io.BytesIO(input_data))
        df = pd.DataFrame(data, columns=['sqft_living'])
        return df
    else:
        raise ValueError(f'Unsupported content type: {content_type}')
        
def predict_fn(input_data, model):
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

    args = parser.parse_args()

    # Read the training data
    train_data_files = [
        os.path.join(args.train, file)
        for file in os.listdir(args.train)
        if file.endswith('.csv')
    ]
    if len(train_data_files) == 0:
        raise ValueError(f'No files found in {args.train}')
    print('Training data files:', train_data_files)
    df_train = pd.concat([pd.read_csv(file) for file in train_data_files])

    # Use 'sqft_living' to predict 'price'
    X_train = df_train[['sqft_living']]
    y_train = df_train['price']

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ensure the model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Save the model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
