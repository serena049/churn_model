# -*- coding: utf-8 -*-

import pathlib as pl
import pandas as pd
import time
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

__author__ = "Wei (Serena) Zou"
__copyright__ = "Wei (Serena) Zou"
__license__ = "mit"


class RawInputChecker:
    """
    Class to check path, data size and data type
    """

    def __init__(self, data_path):
        self.data_path = data_path

    def list_all_files(self) -> pd.DataFrame:
        """
        go through all the files in the data_analysis directory and build a DataFrame with the file name,
        parent path and modified time.
        Returns: a pandas df
        """

        all_files = []
        for i in self.data_path.rglob('*.*'):
            all_files.append((i.name, i.parent, time.ctime(i.stat().st_ctime)))

        columns = ["File_Name", "Parent", "Created"]
        df = pd.DataFrame.from_records(all_files, columns=columns)

        return df

    @staticmethod
    def check_path(df):
        """
        Check if the data folder is empty
        Args:
            df: this is the pandas df listing all the files in the folder
        Returns: raise ValueError is the folder is empty
        """
        if len(df.index) >= 1:
            print("input file exists.")
        else:
            raise ValueError("empty folder!")

    @staticmethod
    def convert(df: pd.DataFrame, file_col_name: str = "File_Name", time_col_name: str = "Created",
                parent_path_col_name: str = "Parent", contains_str: str = "datasets"):
        '''
        This function is used to load the raw csv file, trim whitespaces in numerical columns, and output the file
        to the original folder
        Args:
            df: pandas df with all the files in the directory
            file_col_name: column name of file names in df
            time_col_name: column name of created time in df
            parent_path_col_name: column name of parent path in df
            contains_str: data input file should have this str in the file name
        Returns:
        '''

        # select all the files containing name "datasets"
        files = df[df[file_col_name].str.contains(contains_str, na=False)].reset_index(drop=True)

        # select the most recent data
        file_path = files.sort_values(time_col_name, ascending=False)[parent_path_col_name][0].joinpath(
            files.sort_values(time_col_name, ascending=False)[file_col_name][0])

        # read in raw data
        df = pd.read_csv(file_path)

        # drop cols without headers
        df = df[df.columns.dropna()]

        # import type.yml to specify column types
        yml_file = pl.Path(__file__).resolve().parents[0].joinpath('type.yml')
        with open(yml_file) as f:
            # use safe_load instead load
            type_dic = yaml.safe_load(f)
        col_with_ws = type_dic['Trim_ws']

        # convert white space to nan
        df[col_with_ws] = df[col_with_ws].replace(r'^\s*$', np.nan, regex=True)

        df.to_csv(file_path, index=False)
        print("data conversion done!")

        return

    @staticmethod
    def load(df: pd.DataFrame, file_col_name: str = "File_Name", time_col_name: str = "Created",
             parent_path_col_name: str = "Parent", contains_str: str = "datasets", yml_file: str = 'type.yml',
             name_in_reference: str = 'churn'):

        """
        Find the most recent data file and load the data
        Args:
            df: pandas df with all the files in the directory
            file_col_name: column name of file names in df
            time_col_name: column name of created time in df
            parent_path_col_name: column name of parent path in df
            contains_str: data input file should have this str in the file name
            name_in_reference: the key in type.yml for the corresponding data set
        Returns: pandas df of input data
        """

        # select all the files containing name "datasets"
        files = df[df[file_col_name].str.contains(contains_str, na=False)].reset_index()
        if len(files.index) >= 1:
            # select the most recent data
            file_path = files.sort_values(time_col_name, ascending=False)[parent_path_col_name][0].joinpath(
                files.sort_values(time_col_name, ascending=False)[file_col_name][0])

            # import type.yml to specify column types
            yml_file = pl.Path(__file__).resolve().parents[0].joinpath(yml_file)
            with open(yml_file) as f:
                # use safe_load instead load
                type_dic = yaml.safe_load(f)

            type_reference = type_dic[name_in_reference]

            # load data
            data = pd.read_csv(file_path, header=0, dtype=type_reference)

            # lower case for column names
            data.columns = map(str.lower, data.columns)

            return data
        else:
            raise ValueError("file not exists!")

    @staticmethod
    def check_size(data: pd.DataFrame, n_cols: int = 21):
        """
        Check # of columns and if the data is empty
        Args:
            data: input data
            n_cols: reference of the number of columns
        Returns: pass or raise error
        """
        if (len(data.index) > 0) and (len(data.columns) == n_cols):
            print("data size correct.")
        elif len(data.index) == 0:
            raise ValueError("empty data!")
        else:
            raise ValueError("data size wrong!")


class Transformation:
    """
    Class to conduct data transformation for model inputs
    """

    def __init__(self, df):
        self.df = df

    def remove_missing_value(self, col_na_thres: str = 0.2) -> pd.DataFrame:
        """
        This function is used to remove NAs
        :param col_na_thres: if % of NAs > this threshold, the column will be removed
        :return: df without NAs
        """
        # remove columns with > col_na_thres na
        df_no_na_col = self.df.dropna(thresh=col_na_thres * len(self.df), axis=1)
        # remove rows with any na
        df_no_na_row = df_no_na_col.dropna(how='any')

        return df_no_na_row

    @staticmethod
    def preprocess(df, target_col: str = 'churn'):
        """
        This function is used to preprocess the raw data and make it ready for model fitting.
        :param df: raw data
        :param target_col: target col name
        :return: post-processed df
        """
        df.set_index('customerid', inplace=True)
        df_cat = df.select_dtypes(include='object')
        df_num = df[list(set(df.columns) - set(df_cat.columns))]

        # one-hot encoding for categorical
        if target_col in df_cat.columns:
            df_new_cat = pd.get_dummies(df_cat.drop(target_col, 1))
        else:
            df_new_cat = pd.get_dummies(df_cat)

        # standardize numerical
        std = StandardScaler()
        scaled = std.fit_transform(df_num)
        scaled = pd.DataFrame(scaled, columns=df_num.columns)
        scaled.set_index(df.index, inplace=True)

        # label encode target variable
        if target_col in df_cat.columns:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col])
            target = df[target_col]
            # concat numerical cols and new cat cols
            df_encode = pd.concat([target, scaled, df_new_cat], axis=1)
        else:
            df_encode = pd.concat([scaled, df_new_cat], axis=1)

        print("data ready for modeling!")
        return df_encode


class DataPreprocess:
    def __init__(self, contains_str: str = 'dataset', n_cols: int = 21, data_path_parent_level: int = 2):
        """
        :param contains_str: functions will select files containing the user-specified str, default to "dataset"
        :param n_cols: number of cols the dataset should have
        :param data_path_parent_level: indicates the parent folder is n (default = 2) levels up than the current script
        """
        self.data_path_parent_level = data_path_parent_level
        self.contains_str = contains_str
        self.n_cols = n_cols

    def load_and_encode_data(self):
        # the parent directory of input data path
        input_data_path = pl.Path(__file__).resolve().parents[self.data_path_parent_level].joinpath('data/input/')
        check_data = RawInputChecker(input_data_path)
        df_all_files = check_data.list_all_files()
        check_data.check_path(df_all_files)
        check_data.convert(df_all_files, contains_str=self.contains_str)
        df_raw = check_data.load(df_all_files, contains_str=self.contains_str)
        check_data.check_size(df_raw, n_cols=self.n_cols)

        # encode data
        df_transformation = Transformation(df_raw)
        df_rv_na = df_transformation.remove_missing_value()
        df_encode = df_transformation.preprocess(df_rv_na)

        return df_encode


if __name__ == "__main__":
    data_preprocess = DataPreprocess(data_path_parent_level=2)
    df_encode = data_preprocess.load_and_encode_data()
