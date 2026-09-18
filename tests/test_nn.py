import math
import random
import unittest

from minigrad import Value
from nn import Layer, MLP, Neuron


class TestNN(unittest.TestCase):
    def setUp(self):
        random.seed(0)

    def test_neuron(self):
        neuron = Neuron(3, nonlinear=False)
        output = neuron([Value(1.0), Value(2.0), Value(3.0)])

        self.assertIsInstance(output, Value)
        self.assertEqual(len(neuron.parameters()), 4)  # 3 weights + 1 bias

    def test_layer(self):
        layer = Layer(input_size=2, output_size=3)
        outputs = layer([Value(1.0), Value(-1.0)])

        self.assertEqual(len(outputs), 3)
        self.assertEqual(len(layer.parameters()), 9)  # 3 * (2 weights + 1 bias)

    def test_mlp_shape_and_parameter_count(self):
        model = MLP(input_size=2, layer_sizes=[3, 4, 1])
        outputs = model([Value(1.0), Value(-1.0)])

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(model.parameters()), 30)

    def test_mlp_backward(self):
        model = MLP(input_size=2, layer_sizes=[3, 1])
        prediction = model([Value(0.5), Value(-1.0)])[0]
        loss = (prediction - 1.0) ** 2

        model.zero_grad()
        loss.backward()

        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(math.isfinite(grad) for grad in gradients))
        self.assertTrue(all(grad != 0.0 for grad in gradients))

    def test_final_xor(self):
        # datasets
        inputs = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
        targets = [
            -1.0,
             1.0,
             1.0,
            -1.0,
        ]

        model = MLP(input_size=2, layer_sizes=[2, 4, 1])
        learning_rate = 0.1

        # running 1000 epochs
        for _ in range(1000):

            loss = Value(0.0)
            for i, t in zip(inputs, targets):
                prediction = model(i)[0]
                loss += (prediction - t) ** 2

            loss = loss / len(inputs)

            model.zero_grad()
            loss.backward()
            for parameter in model.parameters():
                parameter.value -= learning_rate * parameter.grad

        for i, t in zip(inputs, targets):
            prediction = model(i)[0].value
            self.assertEqual(prediction > 0, t > 0)


if __name__ == "__main__":
    unittest.main()
