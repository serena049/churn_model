# -*- coding: utf-8 -*-

import pytest
from churn_model.skeleton import fib

__author__ = "Wei (Serena) Zou"
__copyright__ = "Wei (Serena) Zou"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
