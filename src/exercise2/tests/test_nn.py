"""Acceptance tests for Exercise 002. Expected to FAIL until TODOs are filled."""
import random
import pytest

from exercise2.src.dorrigo_learning.foundations.autograd import Value
from dorrigo_learning.foundations.nn import Neuron, Layer, MLP
from dorrigo_learning.foundations.optim import mse, sgd_step


def test_neuron_forward_and_gradient():
    neuron = Neuron(2, activation="linear", rng=random.Random(0))
    neuron.w[0].data = 2.0
    neuron.w[1].data = -1.0
    neuron.b.data = 0.5
    x, y = Value(3.0), Value(4.0)
    z = neuron([x, y])
    assert z.data == pytest.approx(2.5)
    z.backward()
    assert [w.grad for w in neuron.w] == pytest.approx([3.0, 4.0])
    assert neuron.b.grad == pytest.approx(1.0)
    assert [x.grad, y.grad] == pytest.approx([2.0, -1.0])


def test_relu_does_not_pass_negative_gradient():
    neuron = Neuron(1, activation="relu")
    neuron.w[0].data = -2.0
    neuron.b.data = -1.0
    prediction = neuron([3.0])
    assert prediction.data == 0.0
    prediction.backward()
    assert neuron.w[0].grad == 0.0


def test_layer_outputs_and_parameter_count():
    layer = Layer(2, 3, rng=random.Random(3))
    assert len(layer([1.0, -1.0])) == 3
    assert len(layer.parameters()) == 9  # 3 neurons * (2 weights + 1 bias)
    assert len({id(p) for p in layer.parameters()}) == 9


def test_mlp_output_and_parameter_count():
    model = MLP([2, 3, 1], seed=5)
    predictions = model([0.25, 0.75])
    assert len(predictions) == 1
    assert isinstance(predictions[0], Value)
    assert len(model.parameters()) == 13


def test_mse_value_and_gradients():
    a, b = Value(1.0), Value(3.0)
    loss = mse([a, b], [2.0, 1.0])
    assert loss.data == pytest.approx(2.5)
    loss.backward()
    assert [a.grad, b.grad] == pytest.approx([-1.0, 2.0])


def test_mse_rejects_invalid_input():
    with pytest.raises(ValueError):
        mse([], [])
    with pytest.raises(ValueError):
        mse([Value(1.0)], [])


def test_sgd_updates_parameters():
    w = Value(2.0)
    loss = w ** 2
    loss.backward()
    sgd_step([w], 0.1)
    assert w.data == pytest.approx(1.6)
    with pytest.raises(ValueError):
        sgd_step([w], 0)


def test_single_neuron_learns_one_example():
    model = Neuron(1, activation="linear", rng=random.Random(1))
    model.w[0].data = 0.0
    model.b.data = 0.0
    for _ in range(40):
        prediction = model([1.0])
        loss = mse([prediction], [3.0])
        loss.backward()
        sgd_step(model.parameters(), 0.1)
    assert model([1.0]).data == pytest.approx(3.0, abs=0.001)
