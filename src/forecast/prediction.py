import pandas as pd
import pathlib as pl


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


class FcstAndOutputs:
    def __init__(self, df_fcst_encode: pd.DataFrame, data_path_parent_level: int = 2, selected_model=None):
        if selected_model is None:
            raise ValueError("please specify the algorithm used for the forecasts!")
        self.selected_model = selected_model
        self.df_fcst_encode = df_fcst_encode
        self.data_path_parent_level = data_path_parent_level

    def fcst_and_output(self):
        model_fcst = Forecaster(self.df_fcst_encode, self.selected_model).forecast()

        # specify output file paths
        output_data_path = pl.Path(__file__).resolve().parents[self.data_path_parent_level].joinpath('data/output/')
        model_fcst.to_csv(output_data_path.joinpath('forecast_result.csv'))

        return


if __name__ == '__main__':
    print("done!")
