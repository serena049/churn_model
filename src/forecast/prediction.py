from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pathlib as pl
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np


class Forecaster:
    def __init__(self, df_fcst: pd.DataFrame, model):
        """
        :param df_fcst: this is the encoded input df for forecast
        :param model: selected fitted model
        """
        self.df_fcst = df_fcst
        self.model = model

    def forecast(self):
        """
        This function takes in the fitted model and input data and return forecast results
        :return: fcst df
        """
        x_forecast = self.df_fcst.values
        id = self.df_fcst.index
        fcst = pd.DataFrame(self.model.predict_proba(x_forecast), index=id)
        print("forecast done!")
        return fcst


if __name__ == '__main__':
    print("done!")









