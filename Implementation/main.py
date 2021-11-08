import os
import argparse
import pandas as pd
import numpy as np
import warnings
import streamlit as st

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
    
    return pts


def main_wrapper():
    text = ''
    try:
        home_game = 1 if st.session_state.home_game == 'Home' else 0
        b2b = 1 if st.session_state.b2b else 0
        pts = main(st.session_state.player, st.session_state.team, st.session_state.opponent, st.session_state.season_type, home_game, b2b)
        pts = round(pts)
        text = f'Predicted points: {pts}'
    except Exception as e:
        #text = 'Invalid input'
        text = e
        
    st.markdown(f"<h1 style='color:lightgreen'>{text}</h1>", unsafe_allow_html=True)

    
if __name__ == "__main__":
    # Set app title
    st.title("Predict player's points in a game!")

    with st.form('Input Form', clear_on_submit=False):
        # Input data
        st.text_input('Player', key='player', help="Player's name")
        st.text_input('Team', key='team', help="Player's team name")
        st.text_input('Opponent', key='opponent', help="Opponent's team name")
        st.selectbox('Season Type', ['Regular Season', 'Playoffs'], key='season_type', help="Regular Season or Playoff game")
        st.selectbox('Playing', ['Home', 'Away'], key='home_game', help='Home or Away game')
        st.checkbox('Back to Back Game', key='b2b')

        # Submit
        st.form_submit_button('Predict', help='Predict player points!', on_click=main_wrapper)

        
        
# if __name__ == "__main__":
#     # Ignore WARNINGs 
#     warnings.filterwarnings('ignore')
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#     # Create command line parser
#     parser = argparse.ArgumentParser()

#     # Add command line arguments
#     parser.add_argument('--player', '-p', help='<Required> Player Name', required=True)
#     parser.add_argument('--team', '-t', help='<Required> Team Name', required=True)
#     parser.add_argument('--opponent', '-o', help='<Required> Opponent Team Name', required=True)
#     parser.add_argument('--season_type', '-s', help='<Required> Regular Season or Playoffs', required=True)
#     parser.add_argument('--home_game', '-hg', help='<Required> 1 = Home, 0 = Away', required=True)
#     parser.add_argument('--b2b', '-b', help='<Required> 1 = True, 0 = False')

#     # Parse arguments
#     args = parser.parse_args()

#     # Run main
#     main(args.player, args.team, args.opponent, args.season_type, args.home_game, args.b2b)


