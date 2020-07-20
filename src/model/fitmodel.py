from sklearn.model_selection import train_test_split
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
        train_X = train[cols].values
        train_Y = train[target_col].values
        test_X = test[cols].values
        test_Y = test[target_col].values

        print('train, test data split done!')
        return train_X, train_Y, test_X, test_Y


if __name__ == '__main__':
    df_split = AsNumpy(df_encode)
    train_X, train_Y, test_X, test_Y = df_split.train_test_split()