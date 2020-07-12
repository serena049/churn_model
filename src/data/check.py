# -*- coding: utf-8 -*-

# import argparse
# import sys
# import logging

from pathlib import Path

__author__ = "Wei (Serena) Zou"
__copyright__ = "Wei (Serena) Zou"
__license__ = "mit"

# _logger = logging.getLogger(__name__)

class Check():
    """
    Class to check path, data size and data type
    """

    def __init__(self):
        pass

    def check_path(self):
        return

    def check_size(self):
        return

    def check_type(self):
        return

# def fib(n):
#     """Fibonacci example function
#
#     Args:
#       n (int): integer
#
#     Returns:
#       int: n-th Fibonacci number
#     """
#     assert n > 0
#     a, b = 1, 1
#     for i in range(n-1):
#         a, b = b, a+b
#     return a
#
#
# def parse_args(args):
#     """Parse command line parameters
#
#     Args:
#       args ([str]): command line parameters as list of strings
#
#     Returns:
#       :obj:`argparse.Namespace`: command line parameters namespace
#     """
#     parser = argparse.ArgumentParser(
#         description="Just a Fibonacci demonstration")
#     parser.add_argument(
#         "--version",
#         action="version",
#         version="churn_model_project {ver}".format(ver=__version__))
#     parser.add_argument(
#         dest="n",
#         help="n-th Fibonacci number",
#         type=int,
#         metavar="INT")
#     parser.add_argument(
#         "-v",
#         "--verbose",
#         dest="loglevel",
#         help="set loglevel to INFO",
#         action="store_const",
#         const=logging.INFO)
#     parser.add_argument(
#         "-vv",
#         "--very-verbose",
#         dest="loglevel",
#         help="set loglevel to DEBUG",
#         action="store_const",
#         const=logging.DEBUG)
#     return parser.parse_args(args)
#
#
# def setup_logging(loglevel):
#     """Setup basic logging
#
#     Args:
#       loglevel (int): minimum loglevel for emitting messages
#     """
#     logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
#     logging.basicConfig(level=loglevel, stream=sys.stdout,
#                         format=logformat, datefmt="%Y-%m-%d %H:%M:%S")
#
#
# def main(args):
#     """Main entry point allowing external calls
#
#     Args:
#       args ([str]): command line parameter list
#     """
#     args = parse_args(args)
#     setup_logging(args.loglevel)
#     _logger.debug("Starting crazy calculations...")
#     print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
#     _logger.info("Script ends here")
#
#
# def run():
#     """Entry point for console_scripts
#     """
#     main(sys.argv[1:])
#
#
# if __name__ == "__main__":
#     run()
