import os
import argparse
import pandas as pd
import numpy as np
import warnings

from generate_data import generate_features
from model import load_model, predict_points


def main(player, team, opponent, season_type, home_game, b2b=None):
    # Load model
    model_name = season_type.replace(" ", "").lower()
    model = load_model(f"data/nba_predictor_{model_name}")

    # Generate input features for a player
    features = generate_features(player, team, opponent, season_type, home_game, b2b)

    # Predict player points
    pts = predict_points(model, features)
    pts = pts[0][0]

    # Output predicted points
    print("Predicted points: ", pts)


if __name__ == "__main__":
    # Ignore WARNINGs 
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    # Create command line parser
    parser = argparse.ArgumentParser()

    # Add command line arguments
    parser.add_argument('--player', '-p', help='<Required> Player Name', required=True)
    parser.add_argument('--team', '-t', help='<Required> Team Name', required=True)
    parser.add_argument('--opponent', '-o', help='<Required> Opponent Team Name', required=True)
    parser.add_argument('--season_type', '-s', help='<Required> Regular Season or Playoffs', required=True)
    parser.add_argument('--home_game', '-hg', help='<Required> 1 = Home, 0 = Away', required=True)
    parser.add_argument('--b2b', '-b', help='<Required> 1 = True, 0 = False')

    # Parse arguments
    args = parser.parse_args()

    # Run main
    main(args.player, args.team, args.opponent, args.season_type, args.home_game, args.b2b)


