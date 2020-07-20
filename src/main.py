import preprocess.etl as etl
import model.fitmodel as fitmodel
import pathlib as pl
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    # step 1: ETL
    # read in raw data
    input_data_path = pl.Path(__file__).resolve().parents[1].joinpath('data/input/')  # the parent directory of
    check_data = etl.Check(input_data_path)
    df_all_files = check_data.list_all_files()
    check_data.check_path(df_all_files)
    df_raw = check_data.convert(df_all_files)
    df_raw = check_data.load(df_all_files)
    check_data.check_size(df_raw)

    # transformation
    df_transformation = etl.Transformation(df_raw)
    df_rv_na = df_transformation.remove_missing_value()
    df_encode = df_transformation.preprocess(df_rv_na)

    # Step 2: Fit models
    # train test split
    df_split = fitmodel.AsNumpy(df_encode)
    train_x, train_y, test_x, test_y, cols = df_split.train_test_split()

    # run a logistic regression
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)
    fi_imp_sumry, conf_matrix, model_roc_auc, fpr, tpr, thresholds \
        = df_split.churn_prediction(logit, train_x, test_x, train_y, test_y, cols, "coefficients")






