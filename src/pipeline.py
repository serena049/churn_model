# -*- coding: utf-8 -*-


from data.check import Check

import pathlib as pl

__author__ = "Wei (Serena) Zou"
__copyright__ = "Wei (Serena) Zou"
__license__ = "mit"

# _logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # load
    input_data_path = pl.Path(__file__).resolve().parents[2].joinpath(
        'data/input/')  # the parent directory of current script
    print(input_data_path)
    check_data = check.Check(input_data_path)
    # df_all_files = check_data.list_all_files()
    # check_data.check_path(df_all_files)
    # df_raw = check_data.convert(df_all_files)
    # df_raw = check_data.load(df_all_files)
    #
    # # transformation
    # df_transformation = check.Transformation(df_raw)
    # df_rv_na = df_transformation.remove_missing_value()
    #
    # target, df_encode = df_transformation.preprocess(df_rv_na)
    # print(df_encode)
