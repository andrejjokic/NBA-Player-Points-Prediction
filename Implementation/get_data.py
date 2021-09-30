from typing import DefaultDict
import requests
import pandas as pd
import numpy as np


# Teams abbreviations
abbreviations = {
    "ATL":	"Atlanta Hawks",
    "BKN":	"Brooklyn Nets",
    "BOS":	"Boston Celtics",
    "CHA":	"Charlotte Hornets",
    "CHI":	"Chicago Bulls",
    "CLE":	"Cleveland Cavaliers",
    "DAL":	"Dallas Mavericks",
    "DEN":	"Denver Nuggets",
    "DET":	"Detroit Pistons",
    "GSW":	"Golden State Warriors",
    "HOU":	"Houston Rockets",
    "IND":	"Indiana Pacers",
    "LAC":	"LA Clippers",
    "LAL":	"Los Angeles Lakers",
    "MEM":	"Memphis Grizzlies",
    "MIA":	"Miami Heat",
    "MIL":	"Milwaukee Bucks",
    "MIN":	"Minnesota Timberwolves",
    "NOP":	"New Orleans Pelicans",
    "NOH":  "New Orleans Hornets",
    "NYK":	"New York Knicks",
    "OKC":	"Oklahoma City Thunder",
    "ORL":	"Orlando Magic",
    "PHI":	"Philadelphia 76ers",
    "PHX":	"Phoenix Suns",
    "POR":	"Portland Trail Blazers",
    "SAC":	"Sacramento Kings",
    "SAS":	"San Antonio Spurs",
    "TOR":	"Toronto Raptors",
    "UTA":	"Utah Jazz",
    "WAS":	"Washington Wizards"
}

# Necessary Headers for API requests
headers = {
    "Referer": "https://www.nba.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
}


def get_player_points(season="2020-21", season_type="Regular Season", player=None, last_n_games=0):
    url = "https://stats.nba.com/stats/playergamelogs"
    params = {
        "LastNGames": last_n_games,
        "LeagueID": "00",
        "MeasureType": "Base",
        "Month": 0,
        "OpponentTeamID": 0,
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "Totals",
        "Period": 0,
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonType": season_type
    }
    
    response = requests.get(url, headers=headers, params=params).json()

    df = pd.DataFrame(np.array(response["resultSets"][0]["rowSet"]), columns=response["resultSets"][0]["headers"])
    df = df[["SEASON_YEAR", "PLAYER_NAME","TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", 'MIN', "WL", "PTS"]]

    # Filter player if specified
    if player is not None:
        df = df[df["PLAYER_NAME"] == player]

    df['H/A'] = df['MATCHUP'].apply(lambda x: 0 if "@" in x else 1)
    df['MATCHUP'] = df['MATCHUP'].apply(lambda x: abbreviations[x.split()[2]])

    return df


def get_player_stats(season="2020-21", season_type="Regular Season", player=None, last_n_games=0):
    url = "https://stats.nba.com/stats/playergamelogs"
    params = {
        "LastNGames": last_n_games,
        "LeagueID": "00",
        "MeasureType": "Base",
        "Month": 0,
        "OpponentTeamID": 0,
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonType": season_type
    }
    
    response = requests.get(url, headers=headers, params=params).json()

    df = pd.DataFrame(np.array(response["resultSets"][0]["rowSet"]), columns=response["resultSets"][0]["headers"])
    df = df[["SEASON_YEAR", "PLAYER_NAME","TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", 'MIN', "WL", "PTS"]]

    # Filter player if specified
    if player is not None:
        df = df[df["PLAYER_NAME"] == player]

    df['H/A'] = df['MATCHUP'].apply(lambda x: 0 if "@" in x else 1)
    df['MATCHUP'] = df['MATCHUP'].apply(lambda x: abbreviations[x.split()[2]])

    return df


def get_team_opponent_shooting(season="2020-21", season_type="Regular Season", team=None, last_n_games=0):
    url = "https://stats.nba.com/stats/leaguedashteamshotlocations"
    params = {
        "DistanceRange": "5ft Range",
        "LastNGames": last_n_games,
        "LeagueID": "00",
        "MeasureType": "Opponent",
        "Month": 0,
        "OpponentTeamID": 0,
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonType": season_type,
        "TeamID": 0,
        "Location": "",
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "VsConference": "",
        "VsDivision": "",
        "GameSegment": "",
        "GameScope": "",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "StarterBench": ""
    }

    response = requests.get(url, headers=headers, params=params).json()

    df = pd.DataFrame(np.array(response["resultSets"]["rowSet"]), columns=response["resultSets"]["headers"][1]["columnNames"])  
    df = df[["TEAM_NAME", "TEAM_ID", "OPP_FG_PCT"]]
    df = rename_shooting_df_columns(df, response["resultSets"]["headers"][0]["columnNames"], "OPP_FG_PCT")       # Rename columns
    
    # Filter team if specified
    if team is not None:
        df = df[df["TEAM_NAME"] == team]
    
    df['SEASON_YEAR'] = season

    return df


def get_player_shooting(season="2020-21", season_type="Regular Season", player=None, last_n_games=0):
    url = "https://stats.nba.com/stats/leaguedashplayershotlocations"
    params = {
        "College": "",
        "Conference": "",
        "Country": "",
        "DistanceRange": "5ft Range",
        "Division": "",
        "DraftPick": "",
        "DraftYear": "",
        "Height": "",
        "LastNGames": last_n_games,
        "LeagueID": "00",
        "MeasureType": "Base",
        "Month": 0,
        "OpponentTeamID": 0,
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonType": season_type,
        "TeamID": 0,
        "Location": "",
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "VsConference": "",
        "VsDivision": "",
        "GameSegment": "",
        "GameScope": "",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "StarterBench": "",
        "Outcome": ""
    }

    response = requests.get(url, headers=headers, params=params).json()

    df = pd.DataFrame(np.array(response["resultSets"]["rowSet"]), columns=response["resultSets"]["headers"][1]["columnNames"])
    df = df[["PLAYER_NAME", "TEAM_ID", "FGA"]]
    df = rename_shooting_df_columns(df, response["resultSets"]["headers"][0]["columnNames"], "FGA")    # Rename column names

    # Filter player if specified
    if player is not None:
        df = df[df["PLAYER_NAME"] == player]

    df['SEASON_YEAR'] = season

    return df


def get_team_stats(season="2020-21", season_type="Regular Season", team=None, last_n_games=0):
    url = "https://stats.nba.com/stats/leaguedashteamstats"
    params = {
        "Conference": "",
        "Division": "",
        "LastNGames": last_n_games,
        "LeagueID": "00",
        "MeasureType": "Advanced",
        "Month": 0,
        "OpponentTeamID": 0,
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonType": season_type,
        "TeamID": 0,
        "Location": "",
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "VsConference": "",
        "VsDivision": "",
        "GameSegment": "",
        "GameScope": "",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "StarterBench": "",
        "Outcome": "",
        "ShotClockRange": "",
        "TwoWay": 0
    }

    response = requests.get(url, headers=headers, params=params).json()

    df = pd.DataFrame(np.array(response["resultSets"][0]["rowSet"]), columns=response["resultSets"][0]["headers"])
    df = df[["TEAM_NAME", "TEAM_ID", "W_PCT", "OFF_RATING", "DEF_RATING", "PACE"]]

    # Filter player if specified
    if team is not None:
        df = df[df["TEAM_NAME"] == team]
    
    return df


def get_team_results(season="2020-21", season_type="Regular Season", team=None, last_n_games=0):
    url = "https://stats.nba.com/stats/teamgamelogs"
    params = {
        "Conference": "",
        "Division": "",
        "LastNGames": last_n_games,
        "LeagueID": "00",
        "MeasureType": "Advanced",
        "Month": 0,
        "OpponentTeamID": 0,
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "Totals",
        "Period": 0,
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonType": season_type,
        "TeamID": 0,
        "Location": "",
        "SeasonSegment": "",
        "DateFrom": "",
        "DateTo": "",
        "VsConference": "",
        "VsDivision": "",
        "GameSegment": "",
        "GameScope": "",
        "PlayerExperience": "",
        "PlayerPosition": "",
        "StarterBench": "",
        "Outcome": "",
        "ShotClockRange": "",
        "TwoWay": 0
    }

    response = requests.get(url, headers=headers, params=params).json()
    
    df = pd.DataFrame(np.array(response["resultSets"][0]["rowSet"]), columns=response["resultSets"][0]["headers"])
    df = df[["TEAM_NAME", "TEAM_ID", 'SEASON_YEAR', "GAME_ID", "GAME_DATE", 'MATCHUP', "OFF_RATING", "DEF_RATING", "PACE", "WL"]]
    
    # Filter player if specified
    if team is not None:
        df = df[df["TEAM_NAME"] == team]

    df['MATCHUP'] = df['MATCHUP'].apply(lambda x: abbreviations[x.split()[2]])
    
    return df


def rename_shooting_df_columns(df, columns, sufix):
    # Rename column names
    start = len(df.columns) - len(columns)
  
    columns = list(df.columns[0:start]) + [column + " " + sufix for column in columns]
    df.columns = columns

    return df

