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


class AsNumpy:
    def __init__(self, df):
        self.df = df

    def train_test_split(self, target_col: str = 'churn', id_col: str = 'customerid'):
        """
        This function will split the df into train and test data for model fitting and evalution
        :param target_col: target col name
        :param id_col: id col name
        :return: np.arrays for train_x, train_y, test_x, test_y and list of col names of the data used in modeling
        """

        # splitting train and test data
        train, test = train_test_split(self.df, test_size=1 / 4, random_state=42)

        # separating dependent and independent variables
        cols = [i for i in self.df.columns if i not in id_col + target_col]
        train_x = train[cols].values
        train_y = train[target_col].values
        test_x = test[cols].values
        test_y = test[target_col].values

        print('train, test data split done!')
        return train_x, train_y, test_x, test_y, cols


class Model:
    def __init__(self, train_x, train_y, test_x, test_y, prediction_x, cols, algorithm, feature_imp_col):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.cols = cols
        self.algorithm = algorithm
        self.feature_imp_col = feature_imp_col
        self.prediction_x = prediction_x

    def churn_prediction(self):
        """
        This function uses the selected algorithm to fit the model
        :param algorithm: selected algorithm
        :param train_x: np.array of training x
        :param test_x: np.array of testing x
        :param train_y: np.array of training y
        :param test_y: np.array oftesting y
        :param cols: list of cols of the df used for model fitting
        :param feature_imp: str indicating the feature importance variable (coeff. for logistic regression; feature
        importance scores for tree-based models)
        :return: a set of model evaluation metrics
        """

        # train model
        model = self.algorithm.fit(self.train_x, self.train_y)
        predictions = model.predict(self.test_x)
        probabilities = model.predict_proba(self.test_x)
        return model, predictions, probabilities

    def evaluation(self, predictions, probabilities):
        fpr, tpr, thresholds = roc_curve(self.test_y, probabilities[:, 1])
        auc = roc_auc_score(self.test_y, predictions)

        evl = {}
        evl["accuracy"] = accuracy_score(self.test_y, predictions)
        evl["classification report"] = classification_report(self.test_y, predictions)
        evl["confusion materix"] = confusion_matrix(self.test_y, predictions)

        return evl, fpr, tpr, auc

    def feature_importance(self, model):
        # feature_imp: for logistic regression, it is model coefficients, for tree-based models, it is feature
        # importance
        if self.feature_imp_col == "coefficients":
            scores = pd.DataFrame(model.coef_.ravel())
        elif self.feature_imp_col == "features":
            scores = pd.DataFrame(model.feature_importances_)

        # generate feature importance df
        column_df = pd.DataFrame(self.cols)
        fi_imp_sumry = (pd.merge(scores, column_df, left_index=True, right_index=True, how="left"))
        fi_imp_sumry.columns = ["fi_scores", "features"]
        fi_imp_sumry = fi_imp_sumry.sort_values(by="fi_scores", ascending=False)

        return fi_imp_sumry

    @staticmethod
    def roc_curve(fpr, tpr, auc, plt_path, algorithm_name):

        plt.figure(figsize=(12, 16))
        plt.plot(fpr, tpr, linestyle="dotted",
                 color="royalblue", linewidth=2,
                 label="AUC = " + str(np.around(auc, 3)))
        plt.plot([0, 1], [0, 1], linestyle="dashed",
                 color="orangered", linewidth=1.5)
        plt.fill_between(fpr, tpr, alpha=.4)
        plt.fill_between([0, 1], [0, 1], color="k")
        plt.legend(loc="lower right",
                   prop={"size": 12})
        plt.grid(True, alpha=.15)
        plt.title(algorithm_name, color="b")
        plt.xticks(np.arange(0, 1, .3))
        plt.yticks(np.arange(0, 1, .3))
        plt.savefig(plt_path)

        return



if __name__ == '__main__':
    df_split = AsNumpy(df_encode)
    train_x, train_y, test_x, test_y, cols = df_split.train_test_split()

    # run a logistic regression model
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)

    # run a random forest model
    randomforest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                          max_depth=3, max_features='auto', max_leaf_nodes=None,
                                          min_impurity_decrease=0.0, min_impurity_split=None,
                                          min_samples_leaf=1, min_samples_split=2,
                                          min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                                          oob_score=False, random_state=None, verbose=0,
                                          warm_start=False)

    # gaussian NB
    gnb = GaussianNB(priors=None)

    # SVM
    svc_lin = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
                  max_iter=-1, probability=True, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)

    # Graident boosting
    gbtree = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                        criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                        min_impurity_split=None, init=None, random_state=None, max_features=None,
                                        verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated',
                                        validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)

    # fit and compare models
    list_algorithms = [logit, gbtree]
    list_feature_imp_cols = ["coefficients", "features"]
    list_algorithm_names = ['logit', 'gbtree']

    output_data_path = pl.Path(__file__).resolve().parents[1].joinpath('data/output/')
    writer_evl = pd.ExcelWriter(output_data_path.joinpath('model_evaluation.xlsx'), engine='xlsxwriter')
    writer_fi = pd.ExcelWriter(output_data_path.joinpath('model_fi.xlsx'), engine='xlsxwriter')

    for algorithm_name, algorithm, feature_imp_col in zip(list_algorithm_names, list_algorithms, list_feature_imp_cols):
        # instantiate class object
        model = Model(train_x, train_y, test_x, test_y, test_y, cols, algorithm, feature_imp_col)
        # fit model
        model, predictions, probabilities = model.churn_prediction()
        # calculate feature importance
        feature_importance = model.feature_importance(model_fit)
        feature_importance.to_excel(writer_fi, sheet_name=algorithm_name)
        # evaluate model performance
        evl, fpr, tpr, auc = pd.DataFrame([model.evaluation(predictions)])
        pd.DataFrame([evl]).to_excel(writer_evl, sheet_name=algorithm_name)
        # ROC curve
        plt_path = output_data_path.joinpath(algorithm_name+'.png')
        roc_curve(fpr, tpr, auc, plt_path, algorithm_name)


    writer_evl.save()
    writer_fi.save()









