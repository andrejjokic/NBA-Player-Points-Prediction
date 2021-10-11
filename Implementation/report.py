import os
import argparse
import warnings
import pandas as pd
import numpy as np
from evidently.dashboard import Dashboard
from evidently.tabs import RegressionPerformanceTab

from model import load_model, prepare_dataset, predict_points, split_features_and_labels
from generate_data import get_dataset


def main(season_type):
    # Load model and dataset
    model = load_model(f'data/nba_predictor_{season_type}')
    dataset = get_dataset(f'data/nba_dataset_{season_type}.csv')
    dataset = prepare_dataset(dataset)
    features, labels = split_features_and_labels(dataset)

    # Add predictions
    predictions = predict_points(model, features)
    predictions = np.squeeze(np.asarray(predictions))           # Convert Nx1 matrix to array

    # Append predictions and labels
    df = features.copy()
    df['Scored'] = labels
    df['Prediction'] = pd.Series(predictions)
    
    # Column mapping
    column_mapping = {
        'target': 'Scored',
        'prediction': 'Prediction',
        'id': None,
        'numerical_features': features.columns
    }

    # Generate model performance report as HTML file
    model_performance = Dashboard(tabs=[RegressionPerformanceTab])
    model_performance.calculate(df, None, column_mapping=column_mapping)
    model_performance.save(f"data/performance_report_{season_type}.html")


if __name__ == "__main__":
    # Ignore WARNINGs 
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # Create command line parser
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument('--season_type', help='Regular Season or Playoffs')

    # Parse arguments
    args = parser.parse_args()

    # Season type dataset
    season_type = args.season_type.replace(" ", "").lower()

    # Run main
    main(season_type)
