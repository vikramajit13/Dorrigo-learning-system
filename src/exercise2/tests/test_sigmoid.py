import random
import pytest

from src.exercise2.src.dorrigo_learning.foundations.autograd import Value

def test_exp_and_sigmoid():    
    x = Value(0.0)
    y = x.sigmoid()

    y.backward()
    assert abs(y.data - 0.5) < 1e-9
    assert abs(x.grad - 0.25) < 1e-9

    