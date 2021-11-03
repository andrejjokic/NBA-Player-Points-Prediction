import pandas as pd
import numpy as np

from get_data import get_player_points, get_player_shooting, get_team_opponent_shooting, get_team_results, get_team_stats
from datetime import datetime

current_season = "2021-22"


def generate_data(num_of_seasons, season_type, function):
    seasons = ["2020-21", "2019-20", "2018-19", "2017-18", "2016-17", "2015-16", "2014-15", "2013-14", "2012-13", "2011-12", "2010-11"]

    if num_of_seasons > len(seasons):
        num_of_seasons = len(seasons)

    df = pd.DataFrame()

    for i in range(num_of_seasons):
        df = df.append(function(seasons[i], season_type), ignore_index=True)

    df.fillna(0, inplace=True)

    return df


def generate_pg(df, column, group_by, filter_by, last_n_games=5, opp=""):
    def filter(row):
        df_mod = df[(df[group_by] == row[filter_by]) & (df['GAME_DATE'] < row['GAME_DATE'])]
        return df_mod[column].mean() if df_mod.shape[0] > 0 else row[column]

    def filter_last_n(row):
        df_mod = df[(df[group_by] == row[filter_by]) & (df['GAME_DATE'] < row['GAME_DATE'])]
        df_mod.sort_values('GAME_DATE', inplace=True, ascending = False)
        df_mod = df_mod.head(last_n_games)
        return df_mod[column].mean() if df_mod.shape[0] > 0 else row[column]
    
    column1 = f"{opp}{column}_PG"
    column2 = f"{opp}LAST_{last_n_games}_{column}_PG"

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df[column] = pd.to_numeric(df[column])
    df[column1] = df.apply(filter, axis=1)
    df[column2] = df.apply(filter_last_n, axis=1)
    
    return df


def generate_w_pct(df, last_n_games=5, opp=""):
    def filter(row):
        df_mod = df[(df["TEAM_NAME"] == row["TEAM_NAME"]) & (df['GAME_DATE'] < row['GAME_DATE'])]
        df_win = df_mod[df_mod['WL'] == 'W']

        return df_win.shape[0] / df_mod.shape[0] if df_mod.shape[0] > 0 else 0.5

    def filter_last_n(row):
        df_mod = df[(df["TEAM_NAME"] == row["TEAM_NAME"]) & (df['GAME_DATE'] < row['GAME_DATE'])]
        df_mod.sort_values('GAME_DATE', inplace=True, ascending = False)
        df_mod = df_mod.head(last_n_games)
        df_win = df_mod[df_mod['WL'] == 'W']

        return df_win.shape[0] / df_mod.shape[0] if df_mod.shape[0] > 0 else 0.5
    
    column1 = opp + "W_PCT"
    column2 = opp + "LAST_N_W_PCT"

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df[column1] = df.apply(filter, axis=1)
    df[column2] = df.apply(filter_last_n, axis=1)

    return df


def generate_b2b(df):
    def filter(row):
        df_mod = df[(df['PLAYER_NAME'] == row["PLAYER_NAME"]) & (df['GAME_DATE'] < row['GAME_DATE'])]

        if df_mod.shape[0] == 0:
            return 0

        last_game = df_mod['GAME_DATE'].max()
        diff = row['GAME_DATE'] - last_game

        # If games are 2 days in a row
        return 1 if diff.days == 1 else 0

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['B2B'] = df.apply(filter, axis=1)
    
    return df


def generate_dataset(n_seasons=4, season_type="Regular Season", last_n_games=5, dst=None):
    # Generate player points data
    player_points = generate_data(n_seasons, season_type, get_player_points)
    player_points = generate_b2b(player_points)
    player_points = generate_pg(player_points, "PTS", "PLAYER_NAME", "PLAYER_NAME", last_n_games)
    player_points = generate_pg(player_points, "MIN", "PLAYER_NAME", "PLAYER_NAME", 3)

    # Generate player shooting FGA data
    player_shooting = generate_data(n_seasons, season_type, get_player_shooting)

    # Generate team opponent shooting
    team_opp_shooting = generate_data(n_seasons, season_type, get_team_opponent_shooting)

    # Generate team results
    team_results =  generate_data(n_seasons, season_type, get_team_results)
    team_results = generate_w_pct(team_results, last_n_games)
    team_results = generate_pg(team_results, "OFF_RATING", "TEAM_NAME", "TEAM_NAME", last_n_games)
    team_results = generate_pg(team_results, "PACE", "TEAM_NAME", "TEAM_NAME", last_n_games)

    # Add opponent team results
    team_results = generate_w_pct(team_results, last_n_games, "OPP_")
    team_results = generate_pg(team_results, "DEF_RATING", "TEAM_NAME", "MATCHUP", last_n_games, "OPP_")
    team_results = generate_pg(team_results, "PACE", "TEAM_NAME", "MATCHUP", last_n_games, "OPP_")

    # Clean data
    player_points.drop(columns=["GAME_DATE"], inplace=True)
    team_results.drop(columns=["GAME_DATE", "DEF_RATING"], inplace=True)
    team_opp_shooting.drop(columns=["TEAM_ID"], inplace=True)
    team_results.drop(columns=["OFF_RATING", "PACE"], inplace=True)

    # Convert all cells to str, so dfs can be merged
    player_points = player_points.astype(str)
    player_shooting = player_shooting.astype(str)
    team_opp_shooting = team_opp_shooting.astype(str)
    team_results = team_results.astype(str)

    # Merge data
    df = player_points.merge(team_results, on=["SEASON_YEAR", "TEAM_NAME", "GAME_ID", "MATCHUP", "WL"])
    df = df.merge(player_shooting, how="left", on=["PLAYER_NAME", 'TEAM_ID', "SEASON_YEAR"])
    df = df.merge(team_opp_shooting, how="left", left_on=["MATCHUP", "SEASON_YEAR"], right_on=["TEAM_NAME", "SEASON_YEAR"])
    df.drop(columns=["TEAM_NAME_y", "TEAM_ID", "35-39 ft. FGA", "40+ ft. FGA", "35-39 ft. OPP_FG_PCT", "40+ ft. OPP_FG_PCT"], inplace=True)
    df.rename(columns={"TEAM_NAME_x": "TEAM_NAME"}, inplace=True)

    if season_type == "Playoffs":
        df.drop(columns=["B2B"], inplace=True)

    if dst is not None:
        df.to_csv(dst, index=False)

    return df


def get_dataset(src):
    return pd.read_csv(src, dtype=str)


def generate_features(player, team, opponent, season_type, home_game, b2b=None):
    # Return value
    ret = pd.DataFrame()
    
    # Player points
    player_points = get_player_points(current_season, season_type, player)
    last_n_player_points = player_points.head(5)

    # Team stats
    team_stats = get_team_stats(current_season, season_type, team)
    last_n_team_stats = get_team_stats(current_season, season_type, team, 5)

    # Opponent stats
    opp_stats = get_team_stats(current_season, season_type, opponent)
    last_n_opp_stats = get_team_stats(current_season, season_type, opponent, 5)

    # Player shooting
    player_shooting = get_player_shooting(current_season, season_type, player)
    player_shooting.drop(columns=['SEASON_YEAR', 'TEAM_ID', '35-39 ft. FGA', '40+ ft. FGA'], inplace=True)

    # Team opponent shooting
    opponent_shooting = get_team_opponent_shooting(current_season, season_type, opponent)
    opponent_shooting.drop(columns=['SEASON_YEAR', 'TEAM_ID', '35-39 ft. OPP_FG_PCT', '40+ ft. OPP_FG_PCT'], inplace=True)

    # Conversions
    player_points['PTS'] = pd.to_numeric(player_points['PTS'])
    last_n_player_points['PTS'] = pd.to_numeric(last_n_player_points['PTS'])
    player_points['MIN'] = pd.to_numeric(player_points['MIN'])
    last_n_player_points['MIN'] = pd.to_numeric(last_n_player_points['MIN'])

    if b2b is not None: 
        ret.at[0, 'B2B'] = b2b

    ret.at[0, 'H/A'] = home_game
    ret.at[0, 'PTS_PG'] = player_points['PTS'].mean()
    ret.at[0, 'LAST_N_PTS_PG'] = last_n_player_points['PTS'].mean() 
    ret.at[0, 'MIN_PG'] = player_points['MIN'].mean()
    ret.at[0, 'LAST_N_MIN_PG'] = last_n_player_points['MIN'].mean()
    ret.at[0, 'W_PCT'] = team_stats['W_PCT'].iloc[0]
    ret.at[0, 'LAST_N_W_PCT'] = last_n_team_stats['W_PCT'].iloc[0]
    ret.at[0, 'OFF_RATING_PG'] = team_stats['OFF_RATING'].iloc[0]
    ret.at[0, 'LAST_N_OFF_RATING_PG'] = last_n_team_stats['OFF_RATING'].iloc[0]
    ret.at[0, 'PACE_PG'] = team_stats['PACE'].iloc[0]
    ret.at[0, 'LAST_N_PACE_PG'] = last_n_team_stats['PACE'].iloc[0]
    ret.at[0, 'OPP_W_PCT'] = opp_stats['W_PCT'].iloc[0]
    ret.at[0, 'OPP_LAST_N_W_PCT'] = last_n_opp_stats['W_PCT'].iloc[0]
    ret.at[0, 'OPP_DEF_RATING_PG'] = opp_stats['DEF_RATING'].iloc[0]
    ret.at[0, 'OPP_LAST_N_DEF_RATING_PG'] = last_n_opp_stats['DEF_RATING'].iloc[0]
    ret.at[0, 'OPP_PACE_PG'] = opp_stats['PACE'].iloc[0]
    ret.at[0, 'OPP_LAST_N_PACE_PG'] = last_n_opp_stats['PACE'].iloc[0]

    # Append player and team opponent shooting
    ret.at[0, 'PLAYER_NAME'] = player
    ret.at[0, 'TEAM_NAME'] = opponent

    ret = ret.merge(player_shooting)
    ret = ret.merge(opponent_shooting)

    ret.drop(columns=['PLAYER_NAME', 'TEAM_NAME'], inplace=True)
    ret[ret.columns] = ret[ret.columns].apply(pd.to_numeric, errors='coerce')

    return ret
