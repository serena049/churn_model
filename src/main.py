import preprocess.etl as etl
import model.fitmodel as fitmodel
import forecast.prediction as prediction

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
    check_data.convert(df_all_files)
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
    list_algorithm_names = ['logit', 'gbtree', 'rf']
    list_algorithms = [logit, gbtree, randomforest]
    list_feature_imp_cols = ["coefficients", "features", "features"]

    # specify output file paths
    output_data_path = pl.Path(__file__).resolve().parents[1].joinpath('data/output/')
    writer_evl = pd.ExcelWriter(output_data_path.joinpath('model_evaluation.xlsx'), engine='xlsxwriter')
    writer_fi = pd.ExcelWriter(output_data_path.joinpath('model_fi.xlsx'), engine='xlsxwriter')

    # run the models and output results
    models = {}
    for algorithm_name, algorithm, feature_imp_col in zip(list_algorithm_names, list_algorithms, list_feature_imp_cols):

        # instantiate class object
        model = fitmodel.Model(train_x, train_y, test_x, test_y, test_y, cols, algorithm, feature_imp_col)

        # fit model
        model_fit, predictions, probabilities = model.churn_prediction()
        models[algorithm_name] = model_fit
        # calculate feature importance
        feature_importance = model.feature_importance(model_fit)
        feature_importance.to_excel(writer_fi, sheet_name=algorithm_name)
        # evaluate model performance
        evl, fpr, tpr, auc = model.evaluation(predictions, probabilities)
        pd.DataFrame([evl]).to_excel(writer_evl, sheet_name=algorithm_name)
        # ROC curve
        plt_path = output_data_path.joinpath(algorithm_name + '.png')
        model.roc_curve(fpr, tpr, auc, plt_path, algorithm_name)

    writer_evl.save()
    writer_fi.save()

    writer_fi.close()
    writer_evl.close()

    # Step 3: Forecast and output priority list
    # preprocessing

    check_data.convert(df_all_files, contains_str="prediction")
    df_fcst_raw = check_data.load(df_all_files, contains_str="prediction", yml_file="type_fcst.yml")

    check_data.check_size(df_fcst_raw, n_cols=20)

    # transformation
    df_fcst_transformation = etl.Transformation(df_fcst_raw)
    df_fcst_rv_na = df_fcst_transformation.remove_missing_value()
    df_fcst_encode = df_fcst_transformation.preprocess(df_fcst_rv_na)

    selected_model = models['gbtree']

    model_fcst = prediction.Forecast(df_fcst_encode, selected_model)
    fcst_result = model_fcst.forecast()

    fcst_result.to_csv(output_data_path.joinpath('forecast_result.csv'))











