import preprocess.etl as etl
import model.fitmodel as fitmodel
import pathlib as pl

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
    train_X, train_Y, test_X, test_Y = df_split.train_test_split()




