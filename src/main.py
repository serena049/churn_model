import preprocess.etl as etl
import model.fitmodel as fitmodel
import pathlib as pl
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

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
    list_algorithm_names = ['logit', 'gbtree']
    list_algorithms = [logit, gbtree]
    list_feature_imp = ["coefficients", "features"]

    # run the models and output results
    output_data_path = pl.Path(__file__).resolve().parents[1].joinpath('data/output/')
    with pd.ExcelWriter(output_data_path.joinpath(file_name)) as writer:
        for algorithm_name, algorithm, feature_imp in zip(list_algorithm_names, list_algorithms, list_feature_imp):

            # instantiate class object
            model = fitmodel.Model(train_x, train_y, test_x, test_y, cols, algorithm, feature_imp)
            # fit model
            model_fit, predictions = model.churn_prediction()
            # calculate feature importance
            fi_imp_sumry = model.feature_importance(model_fit)
            # evaluate model performance
            evl = model.evaluation(predictions)

            # save model performance and feature importance data to output
            fi_imp_sumry.to_excel(writer,  'feature_importance.xlsx', output_data_path, str(algorithm_name))




            df.to_excel(writer, sheet_name=sheet_name)
        #model.to_excel(pd.DataFrame(evl), 'model_evaluation.xlsx', output_data_path, str(algorithm_name))






