# -*- coding: utf-8 -*-


import argparse
import sys
import logging
import data
import pathlib as pl

from churn_model import __version__

__author__ = "Wei (Serena) Zou"
__copyright__ = "Wei (Serena) Zou"
__license__ = "mit"

_logger = logging.getLogger(__name__)

input_data_path = pl.Path(__file__).resolve().parents[2].joinpath('data/input/') # the parent directory of current script
check_data = data.Check(input_data_path)
df_all_files = check_data.list_all_files()
check_data.check_path(df_all_files)
df_raw = check_data.load(df_all_files)
check_data.check_size(df_raw)

#
# if __name__ == "__main__":
#     run()
