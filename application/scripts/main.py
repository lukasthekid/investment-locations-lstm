import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tensorflow import keras

from model import LSTM_v1, LogisticRegression_v1
from utils import Merger


def plot_results(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig("../../plots/model_performance.png", dpi=300)
    # plt.show()


def evaluate_keras_model(model):
    # Print the best hyperparameters
    print("\n")
    print("LSTM Model")
    y_pred_proba = model.predict(X_test_tf)
    y_pred = np.where(y_pred_proba < 0.5, 0, 1)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Print the Area Under the ROC Curve (AUC-ROC)
    print("AUC-ROC:")
    print(roc_auc_score(y_test, y_pred_proba))


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

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

    # Plot ROC curve
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig("../../plots/model_regression_auc.png", dpi=300)
    # plt.show()

    # Plot feature importance
    model = model.best_estimator_
    feature_importance = np.abs(model.coef_[0])
    feature_names = model.feature_names_in_
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]
    top_n = 10
    top_feature_importance = sorted_feature_importance[:top_n]
    top_feature_names = sorted_feature_names[:top_n]
    fig, ax = plt.subplots()
    ax.barh(top_feature_names, top_feature_importance, color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    ax.set_title('Top 10 Feature Importance')

    plt.tight_layout()
    plt.savefig("../../plots/model_regression_importance.png", dpi=300)

    # plt.show()


def evaluate_top5_and_top10(model_lstm, model_lr, df_model):
    df_fdi = pd.read_csv("../../data/preprocessed/preprocessed_fdi.csv")
    df_fdi = df_fdi[df_fdi['invest_actual'] == 1]
    df_fdi = df_fdi.sample(frac=0.1)
    df_country = pd.read_csv("../../data/preprocessed/preprocessed_country.csv")
    result_lstm = []
    result_lr = []
    scores = {}
    for row in df_fdi.itertuples():
        try:
            df_fdi_filtered = df_fdi[(df_fdi['year'] == row.year) & (df_fdi['cowc_source'] == row.cowc_source) & (
                    df_fdi['isin'] == row.isin)].iloc[0:1]
            # get all countries the companys source country invested to in that year
            df_country_grouped = df_country[
                (df_country['year'] == row.year) & (df_country['cowc_source'] == row.cowc_source)] \
                .groupby('cowc_dest').first().reset_index()
            merger = Merger(df_country=df_country_grouped, df_fdi=df_fdi_filtered,
                            scaler_path="../../data/model/result-scaler.pkl")
            X = merger.get_result()
            df_model_X = df_model.drop(['invest_actual'], axis=1)
            # get all the columns from the model data frame
            cols = set(df_model_X.columns) - set(X.columns)
            for col in cols:
                X[col] = 0
            # only take common columns in same order
            X = X[df_model_X.columns]
            X_test = X.values.reshape((X.shape[0], 1, X.shape[1]))
            y = model_lstm.predict(X_test)
            X_output = X.copy()
            X_output['label'] = y
            output = transform_output(X_output, 10)
            y_true = row.cowc_dest
            top5 = y_true in output['Country'].head(5).values
            top10 = y_true in output['Country'].head(10).values
            result_lstm.append({'top5': top5, 'top10': top10})

            y = model_lr.predict_proba(X)
            X['label'] = y[:, 1]
            output = transform_output(X, 10)
            top5 = y_true in output['Country'].head(5).values
            top10 = y_true in output['Country'].head(10).values
            result_lr.append({'top5': top5, 'top10': top10})
        except ValueError:
            continue

    correct_predictions_top5 = sum([obs['top5'] for obs in result_lstm]) / len(result_lstm)
    correct_predictions_top10 = sum([obs['top10'] for obs in result_lstm]) / len(result_lstm)
    scores['lstm'] = {"top5": correct_predictions_top5, "top10": correct_predictions_top10}

    correct_predictions_top5 = sum([obs['top5'] for obs in result_lr]) / len(result_lr)
    correct_predictions_top10 = sum([obs['top10'] for obs in result_lr]) / len(result_lr)
    scores['lr'] = {"top5": correct_predictions_top5, "top10": correct_predictions_top10}
    print(scores)


def calculate_top_score(result_lstm: list, result_lr: list):
    correct_predictions_top5 = sum([obs['top5'] for obs in result_lstm])


def transform_output(df_raw: pd.DataFrame, n: int):
    # Backtransform 'cowc_dest' one-hot encoded columns
    cols = [col for col in df_raw.columns if 'cowc_dest' in col]
    df_subset = df_raw[cols]
    # Back-transform the one-hot encoded values
    df_raw['Country'] = df_subset.idxmax(axis=1).str.replace('cowc_dest_', '')
    df = df_raw.sort_values('label', ascending=False).reset_index(drop=True)
    # df['Country'] = df.apply(lambda x: get_country_name(str(df['Country'])))
    return df[['Country', 'label']].head(n)


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

    print("SkLearn Model saved and finished training")


if __name__ == '__main__':
    # define data
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'keras-tuner'])
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", Warning)

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

    # start training
    # start_model_training()

    # evaluate
    model_lstm = keras.models.load_model("../../data/model/lstm.keras")
    with open('../../data/model/log_reg_model.pkl', 'rb') as f:
        model_lr = pickle.load(f)

    # evaluate_keras_model(model_lstm)
    # evaluate_sklearn_model(model_lr)
    # evaluate_top5_and_top10(model_lstm, model_lr, df)
