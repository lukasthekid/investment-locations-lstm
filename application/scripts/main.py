import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import matplotlib.pyplot as plt
import pickle
from model import LSTM_v1, LogisticRegression_v1
from tensorflow import keras


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
    plt.savefig("../../plots/model_performance.png", dpi=300)
    # plt.show()

def evaluate_keras_model(model):
    # Print the best hyperparameters
    print("\n")
    print("LSTM Model")
    y_pred = model.predict(X_test_tf)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def evaluate_sklearn_model(model):
    # Print the best hyperparameters
    print("\n")
    print("Logitic Regression Model")
    print('Best Penalty:', model.best_estimator_.get_params()['penalty'])
    print('Best C:', model.best_estimator_.get_params()['C'])
    # Use the model to make predictions on the test set
    y_pred = model.predict(X_test)
    # Predict the probabilities of the test data
    y_pred_proba = model.predict_proba(X_test)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Print the Area Under the ROC Curve (AUC-ROC)
    print("AUC-ROC:")
    print(roc_auc_score(y_test, y_pred_proba[:, 1]))

    # Plot feature importance
    model = model.best_estimator_
    feature_importance = np.abs(model.coef_[0])
    feature_names = model.feature_names_in_
    fig, ax = plt.subplots()
    ax.barh(feature_names, feature_importance, color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance')

    plt.tight_layout()
    plt.savefig("../../plots/model_regression_importance.png", dpi=300)

    #plt.show()



def start_model_training():
    lstm = LSTM_v1(X_train_tf, X_val_tf, y_train_tf, y_val_tf)
    best_model = lstm.get_best_model()
    history = best_model.fit(X_train_tf, y_train_tf, epochs=20, batch_size=None, validation_data=(X_val_tf, y_val_tf),
                             verbose=2,
                             shuffle=False)

    plot_results(history)
    best_model.save("../../data/model/lstm.keras")
    print("Keras Model saved and finished training \n \n")
    # evaluate_keras_model(best_model)

    # build logistic regression model
    reg_model = LogisticRegression_v1(X_train, X_test, y_train, y_test)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'penalty': ['l2', None]}
    reg_model = reg_model.build_tuned_model(param_grid)

    evaluate_sklearn_model(reg_model)
    # Save the model to a file
    with open('../../data/model/log_reg_model.pkl', 'wb') as f:
        pickle.dump(reg_model, f)

    print("ScLearn Model saved and finished training")




if __name__ == '__main__':
    #define data
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'keras-tuner'])
    warnings.simplefilter("ignore", UserWarning)

    df = pd.read_csv("../../data/preprocessed/preprocessed_data.csv")
    # basic split for sklearn models
    X_train, X_test, y_train, y_test = train_test_split(df.drop('invest_actual', axis=1), df['invest_actual'],
                                                        test_size=0.33, random_state=42, stratify=df['invest_actual'])

    # Further splitting for the keras models
    X_train_tf, X_val_tf, y_train_tf, y_val_tf = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
                                                                  stratify=y_train)

    # Reshape input to be 3D [samples, timesteps, features]
    X_train_tf = X_train_tf.values.reshape((X_train_tf.shape[0], 1, X_train_tf.shape[1]))
    X_val_tf = X_val_tf.values.reshape((X_val_tf.shape[0], 1, X_val_tf.shape[1]))
    X_test_tf = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    #start training
    #start_model_training()
    #evaluate
    model_lstm = keras.models.load_model("../../data/model/lstm.keras")
    with open('../../data/model/log_reg_model.pkl', 'rb') as f:
        model_lr = pickle.load(f)

    #evaluate_keras_model(model_lstm)
    evaluate_sklearn_model(model_lr)
