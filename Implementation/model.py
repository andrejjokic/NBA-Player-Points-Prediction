import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
import keras_tuner as kt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from generate_data import get_dataset


def make_np_print_prettier():
    # Make numpy printouts easier to read.
    np.set_printoptions(precision=3, suppress=True)


def prepare_dataset(df, min_minutes=23):
    # Clean data
    df = df.drop(columns=['SEASON_YEAR', 'PLAYER_NAME', 'TEAM_NAME', 'GAME_ID', 'MATCHUP', 'WL'])
    df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')

    # Drop stats with players who played under min_minutes
    df = df[df['MIN'] > min_minutes]

    # Drop any stil NaN information
    df.dropna(inplace=True)

    return df


def remove_outliers(df, frac=0.45):
    return df[abs(df['PTS'] - (df['PTS_PG'] + df['LAST_N_PTS_PG']) / 2) < ((df['PTS_PG'] + df['LAST_N_PTS_PG']) * (0.5 * frac))]


def split_dataset(df, train_pct=0.8): 
    # Split training and test data
    train = df.sample(frac=train_pct, random_state=0)
    test = df.drop(train.index)

    return train, test


def split_features_and_labels(df):
    # Split features and labels
    df_x = df.drop(columns=['PTS', 'MIN'])
    df_y = df['PTS']

    return df_x, df_y


def create_model(train_x):
    # Input layer
    input_layer = preprocessing.Normalization(axis=-1)
    input_layer.adapt(np.array(train_x))

    # Build model
    model = keras.Sequential([
        input_layer,
        layers.Dense(units=368, activation='relu'),
        layers.Dense(1)
    ])

    # Compile model
    model.compile(
        loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(0.001)
    )

    return model


def train_model(model, train_x, train_y, validation_split=0.2, verbose=0, epochs=100):
    history = model.fit(
        train_x, 
        train_y,
        validation_split=validation_split,
        verbose=verbose,
        epochs=epochs
    )

    return history, model


def test_model(model, test_x, test_y, verbose=0):
    return model.evaluate(test_x, test_y, verbose)


def save_model(model, dst):
    model.save(dst)


def load_model(src):
    return tf.keras.models.load_model(src)


def predict_points(model, features):
    return model.predict(features)


def hypertune(train_x, train_y):
    # Select the right set of hyperparameters

    def model_builder(hp):
        # Model builder function returns a compiled model and uses hyperparameters defined inline to hypertune the model        

        # Input layer
        input_layer = preprocessing.Normalization(axis=-1)
        input_layer.adapt(np.array(train_x))

        # Tune the number of units and activation function in hidden layer
        hp_units = hp.Int('units', min_value=16, max_value=512, step=32)
        hp_activation = hp.Choice("activation", values=['relu', 'sigmoid', 'tanh']) 

        # Build model
        model = keras.Sequential([
            input_layer,
            layers.Dense(units=hp_units, activation=hp_activation),
            layers.Dense(1)
        ])

        # Tune learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-2, 1e-1])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=['mean_absolute_error']
        )

        return model

    # Instantiate hyper tuner
    tuner = kt.Hyperband(
        model_builder,
        objective='val_mean_absolute_error',
        max_epochs=10,
        factor=3,
        directory='hypertune_dir',
        project_name='hypertune'
    )

    # A callback to stop training early after reaching a certain value of the validation loss
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Run the hyperparameters search
    tuner.search(train_x, train_y, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
            The hyperparameter search is complete. 
            The optimal number of units in the first densely-connected layer is {best_hps.get('units')}.
            The optimal activation function in the first densely-connected layer is {best_hps.get('activation')}
            The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
        """
    )


def main(dataset, dst):
    df = get_dataset(dataset)
    df = prepare_dataset(df)

    train, test = split_dataset(df)

    train_x, train_y = split_features_and_labels(train)
    test_x, test_y = split_features_and_labels(test)

    model = create_model(train_x)
    history, model = train_model(model, train_x, train_y)

    mae = test_model(model, test_x, test_y)
    print("MAE on test: ", mae)

    save_model(model, dst)
    

if __name__ == "__main__":
    main("data/nba_dataset_playoffs.csv", "data/nba_predictor_playoffs")
    main("data/nba_dataset_regularseason.csv", "data/nba_predictor_regularseason")