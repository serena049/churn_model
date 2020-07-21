from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score


class AsNumpy:
    def __init__(self, df):
        self.df = df

    def train_test_split(self, target_col: str = 'churn', id_col: str = 'customerid'):

        # splitting train and test data
        train, test = train_test_split(self.df, test_size=1/4, random_state=42)

        # separating dependent and independent variables
        cols = [i for i in self.df.columns if i not in id_col + target_col]
        train_x = train[cols].values
        train_y = train[target_col].values
        test_x = test[cols].values
        test_y = test[target_col].values

        print('train, test data split done!')
        return train_x, train_y, test_x, test_y, cols

    @staticmethod
    def churn_prediction(algorithm, training_x, testing_x, training_y, testing_y, cols: list, feature_imp: str):

        # model
        global fi_scores
        algorithm.fit(training_x, training_y)
        predictions = algorithm.predict(testing_x)
        probabilities = algorithm.predict_proba(testing_x)

        # feature_imp: for logistic regression, it is model coefficients, for tree-based models, it is feature
        # importance
        if feature_imp == "coefficients":
            fi_scores = pd.DataFrame(algorithm.coef_.ravel())
        elif feature_imp == "features":
            fi_scores = pd.DataFrame(algorithm.feature_importances_)

        # generate feature importance df
        column_df = pd.DataFrame(cols)
        fi_imp_sumry = (pd.merge(fi_scores, column_df, left_index=True, right_index=True, how="left"))
        fi_imp_sumry.columns = ["fi_scores", "features"]
        fi_imp_sumry = fi_imp_sumry.sort_values(by="fi_scores", ascending=False)

        # print model performance
        print(algorithm)
        print("\n Classification report : \n", classification_report(testing_y, predictions))
        print("Accuracy   Score : ", accuracy_score(testing_y, predictions))

        # confusion matrix
        conf_matrix = confusion_matrix(testing_y, predictions)
        # roc_auc_score
        model_roc_auc = roc_auc_score(testing_y, predictions)
        print("Area under curve : ", model_roc_auc, "\n")
        fpr, tpr, thresholds = roc_curve(testing_y, probabilities[:, 1])

        return fi_imp_sumry, conf_matrix, model_roc_auc, fpr, tpr, thresholds



if __name__ == '__main__':
    df_split = AsNumpy(df_encode)
    train_x, train_y, test_x, test_y, cols = df_split.train_test_split()
    # run a logistic regression
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)
    fi_imp_sumry, conf_matrix, model_roc_auc, fpr, tpr, thresholds \
        = df_split.churn_prediction(logit, train_x, test_x, train_y, test_y, cols, "coefficients")

