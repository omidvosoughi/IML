from testbook import testbook
import pytest
import socket
import numpy as np


@pytest.fixture
def tb():
    with testbook('exercise.ipynb') as tb:
        yield tb
    
    
def test_1_1(tb):
    tb.execute_cell([1,4])
        
    X = tb.ref('X')
    y = tb.ref('y')
        
    assert X.shape == (199, 7)
    assert y.shape == (199,)
        
    # Make sure array is shuffled
    assert X.tolist()[0][0] == pytest.approx(12.72, 0)

        
def test_1_2(tb):
    tb.execute_cell([1,4,6])
        
    X = tb.ref('X')
        
    assert not np.isnan(X.tolist()).any()
        
    X_flattend = X.flatten().tolist()
    X_min = min(X_flattend)
    X_max = max(X_flattend)
        
    assert 0 == pytest.approx(X_min, 0)
    assert 1 == pytest.approx(X_max, 0)
        
    y = tb.ref('y')
    y_flattend = y.flatten().tolist()
    
    assert np.isnan(y_flattend).any() == False
    

def test_1_3(tb):
    tb.execute_cell([1,4,6,8])
        
    y_train = tb.ref('y_train')
    y_val = tb.ref('y_val')

    assert y_train.shape == (119,)
    assert y_val.shape == (80,)
    

def test_2_1(tb):
    tb.execute_cell([1,4,6,8,11])
    
    get_sequential_model = tb.ref('get_sequential_model')

    model = get_sequential_model(
        num_hidden_layers=0,
        num_hidden_units=5,
        input_units=9,
        output_units=1)
    modules_string = str(model._modules)
    
    assert modules_string.count("Linear") == 2
    assert modules_string.count("ReLU") == 1
    assert modules_string.count("in_features=9") == 1
    assert modules_string.count("out_features=5") == 1
    assert modules_string.count("in_features=5") == 1
    assert modules_string.count("out_features=1") == 1

    assert len(model) == 3

    model = get_sequential_model(
        num_hidden_layers=2,
        num_hidden_units=3,
        input_units=512,
        output_units=40)
    modules_string = str(model._modules)
    
    assert modules_string.count("Linear") == 4
    assert modules_string.count("ReLU") == 3
    assert modules_string.count("in_features=512") == 1
    assert modules_string.count("out_features=3") == 3
    assert modules_string.count("in_features=3") == 3
    assert modules_string.count("out_features=40") == 1

    assert len(model) == 7


def test_2_2(tb):
    tb.execute_cell([1,4,6,8,11,12,14])
    
    input_units = tb.ref('input_units')
    output_units = tb.ref('output_units')
    
    assert input_units == 7
    assert output_units == 3


def test_3(tb):
    tb.execute_cell([1,4,6,8,11,12,14,17,19])
    
    def to_list(str, cast=int):
        str = str.replace('[', '')
        str = str.replace(']', '')
        str = str.replace('"', '')
        str = str.replace('\'', '')
        
        entries = str.split(", ")
        return [cast(entry) for entry in entries]
    
    x = to_list(str(tb.ref('x')), int)
    y = to_list(str(tb.ref('y')), float)
    highest_accuracy = float(tb.ref('highest_accuracy'))
    lowest_accuracy = float(tb.ref('lowest_accuracy'))

    # Test monotomy
    for x1, x2 in zip(x, x[1:]):
        assert x1 <= x2

    for y1, y2 in zip(y, y[1:]):
        assert y1 <= y2
    
    # Test min+max values
    assert min(y) == pytest.approx(0.387, 0.01)
    assert max(y) == pytest.approx(0.949, 0.01)
    assert highest_accuracy == pytest.approx(0.949, 0.01)
    assert lowest_accuracy == pytest.approx(0.300, 0.01)
    

if __name__ == "__main__":
    with testbook('solution.ipynb') as tb:
        test_1_1(tb)
        test_1_2(tb)
        test_1_3(tb)
        test_2_1(tb)
        test_2_2(tb)
        test_3(tb)


