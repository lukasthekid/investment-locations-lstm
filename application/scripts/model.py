import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import save_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import Objective
import sys
import subprocess
from kerastuner.tuners import RandomSearch
import matplotlib.pyplot as plt
from keras import regularizers
from keras.callbacks import EarlyStopping


class LSTM_v1:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.model = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def build_model(self, hp):
        model = keras.Sequential()
        model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                              input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                              kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(1, activation=hp.Choice('dense_activation', values=['relu', 'sigmoid'], default='sigmoid')))
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(
                hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(),
                     keras.metrics.BinaryAccuracy()])  # mean accuracy rate across all predictions for binary classification problems.
        return model

    def get_best_model(self):
        tuner = RandomSearch(
            self.build_model,
            objective=Objective("auc", direction="max"),
            max_trials=10,
            executions_per_trial=2)

        tuner.search(self.X_train, self.y_train,
                     epochs=5,
                     batch_size=128,
                     validation_data=(self.X_test, self.y_test))

        best_model = tuner.get_best_models(num_models=1)[0]
        self.model = best_model
        return best_model
