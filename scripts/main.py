import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import sys
import subprocess

import matplotlib.pyplot as plt

from scripts.model import LSTM_v1


def plot_results(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.savefig("../plots/model_performance.png", dpi=300)
    # plt.show()


if __name__ == '__main__':
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'keras-tuner'])
    warnings.simplefilter("ignore", UserWarning)

    df = pd.read_csv("../data/preprocessed/preprocessed_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.drop('invest_actual', axis=1), df['invest_actual'],
                                                        test_size=0.33, random_state=42, stratify=df['invest_actual'])

    # Reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm = LSTM_v1(X_train, X_test, y_train, y_test)
    best_model = lstm.get_best_model()
    history = best_model.fit(X_train, y_train, epochs=50, batch_size=None, validation_data=(X_test, y_test), verbose=2,
                             shuffle=False)

    plot_results(history)
    best_model.save("../data/model/lstm.keras")
    # y_pred = best_model.predict(X_test)
    # print(y_pred)
