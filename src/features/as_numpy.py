# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score




class TrainTestSplit:

    # splitting train and test data
    def __init__(self, data, target_col: str='churn', id_col: str='customerid'):

        self.data = data
        self.target_col = target_col
        self.id_col = id_col

    def train_test_split(self):
        train, test = train_test_split(self.data, test_size=0.25, random_state=111)

        # seperating dependent and independent variables
        all_cols = self.data.columns
        id_col = self.id_col
        target_col = self.target_col
        cols = [i for i in all_cols if i not in id_col + target_col]
        train_x = train[cols].values
        train_y = train[target_col].values
        test_x = test[cols].values
        test_y = test[target_col].values

        return train_x, train_y, test_x, test_y, cols


    @staticmethod
    def churn_prediction(self, algorithm, training_x, testing_x, training_y, testing_y, cols, cf, threshold_plot):

        # model
        algorithm.fit(training_x, training_y)
        predictions = algorithm.predict(testing_x)
        probabilities = algorithm.predict_proba(testing_x)

        # coeffs (coef for logistic regression , feature importance for tree models)
        if cf == "coefficients":
            coefficients = pd.DataFrame(algorithm.coef_.ravel())
        elif cf == "features":
            coefficients = pd.DataFrame(algorithm.feature_importances_)
        column_df = pd.DataFrame(cols)
        coef_summary = (pd.merge(coefficients, column_df, left_index=True, right_index=True, how="left"))
        coef_summary.columns = ["coefficients", "features"]
        coef_summary = coef_summary.sort_values(by="coefficients", ascending=False)

        # evaluation
        report = classification_report(testing_y, predictions)
        accuracy = accuracy_score(testing_y, predictions)

        # confusion matrix
        conf_matrix = confusion_matrix(testing_y, predictions)

        # roc_auc_score
        model_roc_auc = roc_auc_score(testing_y, predictions)
        fpr, tpr, thresholds = roc_curve(testing_y, probabilities[:, 1])

        return coef_summary, report, accuracy, conf_matrix, model_roc_auc,fpr, tpr, thresholds