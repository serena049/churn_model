# -*- coding: utf-8 -*-

import argparse
import sys
import logging
import pathlib as pl
import pandas as pd
import time
import numpy as np
import yaml

__author__ = "Wei (Serena) Zou"
__copyright__ = "Wei (Serena) Zou"
__license__ = "mit"

# _logger = logging.getLogger(__name__)

class Check:
    """
    Class to check path, data size and data type
    """
    #
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
    def load(df: pd.DataFrame, file_col_name: str = "File_Name", time_col_name: str = "Created",
             parent_path_col_name: str = "Parent", contains_str: str = "datasets", name_in_reference: str = 'churn'):

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
        files = df[df[file_col_name].str.contains(contains_str, na=False)]
        if len(files.index) >= 1:
            # select the most recent data
            file_path = files.sort_values(time_col_name, ascending=False)[parent_path_col_name][0].joinpath(
                        files.sort_values(time_col_name, ascending=False)[file_col_name][0])

            # import type.yml to specify column types
            yml_file = pl.Path(__file__).resolve().parents[0].joinpath('type.yml')
            with open(yml_file) as f:
                # use safe_load instead load
                type_reference = yaml.safe_load(f)[name_in_reference]

            # load data
            data = pd.read_csv(file_path, header=0, names=type_reference.keys(), dtype=type_reference)

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
        else:
            raise ValueError("data size wrong!")


if __name__ == "__main__":
    input_data_path = pl.Path(__file__).resolve().parents[2].joinpath('data/input/') # the parent directory of current script
    check_data = Check(input_data_path)
    df_all_files = check_data.list_all_files()
    check_data.check_path(df_all_files)
    df_raw = check_data.load(df_all_files)
    check_data.check_size(df_raw)
    print(df_raw.head())



