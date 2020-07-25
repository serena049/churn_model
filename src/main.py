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
    data_preprocess = etl.DataPreprocess()
    df_encode = data_preprocess.load_and_encode_data()

    # Step 2: Fit models
    list_of_models = ['logit', 'gbtree', 'rf']     # user can modify
    train_models = fitmodel.FitModel(list_of_models=list_of_models)
    train_models.train_models()

    # Step 3: Forecast and output priority list
    check_data.convert(df_all_files, contains_str="prediction")
    df_fcst_raw = check_data.load(df_all_files, contains_str="prediction", yml_file="type_fcst.yml")

    check_data.check_size(df_fcst_raw, n_cols=20)

    # transformation
    df_fcst_transformation = etl.Transformation(df_fcst_raw)
    df_fcst_rv_na = df_fcst_transformation.remove_missing_value()
    df_fcst_encode = df_fcst_transformation.preprocess(df_fcst_rv_na)

    selected_model = models['gbtree']

    model_fcst = prediction.Forecaster(df_fcst_encode, selected_model)
    fcst_result = model_fcst.forecast()

    fcst_result.to_csv(output_data_path.joinpath('forecast_result.csv'))











