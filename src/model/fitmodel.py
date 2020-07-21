from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pathlib as pl
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
    def __init__(self, train_x, train_y, test_x, test_y, prediction_x, cols, algorithm, feature_imp):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.cols = cols
        self.algorithm = algorithm
        self.feature_imp = feature_imp

    def churn_prediction(self):
        """
        This function uses the selected algorithm to fit the model
        :param algorithm: selected algorithm
        :param training_x: np.array of training x
        :param testing_x: np.array of testing x
        :param training_y: np.array of training y
        :param testing_y: np.array oftesting y
        :param cols: list of cols of the df used for model fitting
        :param feature_imp: str indicating the feature importance variable (coeff. for logistic regression; feature
        importance scores for tree-based models)
        :return: a set of model evaluation metrics
        """

        # train model
        model = self.algorithm.fit(self.train_x, self.train_y)
        predictions = model.predict(self.test_x)
        return model, predictions

    def prediction(self, model, top_n: int = 30):
        probabilities = model.predict_proba(self.test_x).sort_values(ascending=False)
        return probabilities

    def evaluation(self, predictions):
        evl = {}
        evl["accuracy"] = accuracy_score(self.test_y, predictions)
        evl["classification report"] = classification_report(self.test_y, predictions)
        evl["confusion materix"] = confusion_matrix(self.test_y, predictions)
        evl["auc"] = roc_auc_score(self.test_y, predictions)
        return evl

    def feature_importance(self, model):
        # feature_imp: for logistic regression, it is model coefficients, for tree-based models, it is feature
        # importance
        if self.feature_imp == "coefficients":
            scores = pd.DataFrame(model.coef_.ravel())
        elif self.feature_imp == "features":
            scores = pd.DataFrame(model.feature_importances_)

        # generate feature importance df
        column_df = pd.DataFrame(self.cols)
        fi_imp_sumry = (pd.merge(scores, column_df, left_index=True, right_index=True, how="left"))
        fi_imp_sumry.columns = ["fi_scores", "features"]
        fi_imp_sumry = fi_imp_sumry.sort_values(by="fi_scores", ascending=False)

        return fi_imp_sumry

    @staticmethod
    def to_excel(df, file_name, file_path, sheet_name):
        with pd.ExcelWriter(file_path.joinpath(file_name)) as writer:
            df.to_excel(writer, sheet_name=sheet_name)
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
    list_feature_imp = ["coefficients", "features"]
    list_algorithm_names = ['logit', 'gbtree']

    all_evl = np.array()
    all_feature_importance = np.array()

    for algorithm_name, algorithm, feature_imp in zip(list_algorithm_names, list_algorithms, list_feature_imp):
        # instantiate class object
        model = Model(train_x, train_y, test_x, test_y, cols, algorithm, feature_imp)
        # fit model
        model_fit, predictions = model.churn_prediction()
        # calculate feature importance
        fi_imp_sumry = model.feature_importance(model_fit)
        # evaluate model performance
        evl = model.evaluation(predictions)
        # save model performance and feature importances
        output_data_path = pl.Path(__file__).resolve().parents[2].joinpath('data/output/')
        model.to_excel(fi_imp_sumry, 'feature_importance.xlsx', output_data_path, str(algorithm))
        model.to_excel(evl, 'model_evaluation.xlsx', output_data_path, str(algorithm))
