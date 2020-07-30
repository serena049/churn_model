import preprocess.etl as etl
import model.fitmodel as fitmodel
import forecast.prediction as prediction

if __name__ == "__main__":
    # step 1: ETL
    df_encode = etl.DataPreprocess().load_and_encode_data()
    # Step 2: Fit models
    list_of_models = ['logit', 'gbtree', 'rf']     # user can modify
    train_models = fitmodel.FitModel(df_encode, list_of_models=list_of_models)
    models = train_models.train_models()

    # Step 3: Forecast and output priority list
    df_fcst_encode = etl.DataPreprocess(contains_str='prediction', n_cols=20).load_and_encode_data()
    selected_model = models['gbtree']
    prediction.FcstAndOutputs(df_fcst_encode, selected_model=selected_model).fcst_and_output()










