import math
import unittest

from minigrad import Value


def central_difference(function, x, epsilon=1e-6):
    """Approximate df/dx using values immediately around x."""
    return (function(x + epsilon) - function(x - epsilon)) / (2 * epsilon)


class TestValueOperations(unittest.TestCase):
    def test_addition(self):
        x = Value(2.0)
        y = Value(3.0)
        result = x + y

        result.backward()

        self.assertEqual(result.value, 5.0)
        self.assertEqual(x.grad, 1.0)
        self.assertEqual(y.grad, 1.0)

    def test_multiplication(self):
        x = Value(2.0)
        y = Value(3.0)
        result = x * y

        result.backward()

        self.assertEqual(result.value, 6.0)
        self.assertEqual(x.grad, 3.0)
        self.assertEqual(y.grad, 2.0)

    def test_negation_and_subtraction(self):
        x = Value(5.0)
        y = Value(2.0)
        result = x - y

        result.backward()

        self.assertEqual(result.value, 3.0)
        self.assertEqual(x.grad, 1.0)
        self.assertEqual(y.grad, -1.0)

    def test_power_with_negative_base(self):
        x = Value(-2.0)
        result = x**2

        result.backward()

        self.assertEqual(result.value, 4.0)
        self.assertEqual(x.grad, -4.0)

    def test_exp(self):
        x = Value(0.7)
        result = x.exp()

        result.backward()

        expected = math.exp(0.7)
        self.assertAlmostEqual(result.value, expected)
        self.assertAlmostEqual(x.grad, expected)

    def test_silu(self):
        x = Value(2.0)
        result = x.silu()

        result.backward()

        sigmoid = 1 / (1 + math.exp(-2.0))
        expected_value = 2.0 * sigmoid
        expected_gradient = sigmoid + 2.0 * sigmoid * (1 - sigmoid)
        self.assertAlmostEqual(result.value, expected_value)
        self.assertAlmostEqual(x.grad, expected_gradient)

    def test_branching_graph_accumulates_gradients(self):
        x = Value(3.0)
        result = x * x + x

        result.backward()

        self.assertEqual(result.value, 12.0)
        self.assertEqual(x.grad, 7.0)

    def test_python_scalars_on_both_sides(self):
        cases = [
            ("x + 2", lambda x: x + 2, 6.0, 1.0),
            ("2 + x", lambda x: 2 + x, 6.0, 1.0),
            ("x - 2", lambda x: x - 2, 2.0, 1.0),
            ("2 - x", lambda x: 2 - x, -2.0, -1.0),
            ("x * 2", lambda x: x * 2, 8.0, 2.0),
            ("2 * x", lambda x: 2 * x, 8.0, 2.0),
            ("x / 2", lambda x: x / 2, 2.0, 0.5),
            ("2 / x", lambda x: 2 / x, 0.5, -0.125),
        ]

        for name, operation, expected_value, expected_gradient in cases:
            with self.subTest(name=name):
                x = Value(4.0)
                result = operation(x)
                result.backward()

                self.assertAlmostEqual(result.value, expected_value)
                self.assertAlmostEqual(x.grad, expected_gradient)


class TestNumericalGradients(unittest.TestCase):
    def test_gradients_match_central_finite_differences(self):
        def numeric_silu(x):
            return x / (1 + math.exp(-x))

        cases = [
            ("addition", lambda x: x + 2, lambda x: x + 2, 1.3),
            ("subtraction", lambda x: 2 - x, lambda x: 2 - x, 1.3),
            ("multiplication", lambda x: x * 3, lambda x: x * 3, 1.3),
            ("power", lambda x: x**3, lambda x: x**3, 1.3),
            ("division", lambda x: 2 / x, lambda x: 2 / x, 1.3),
            ("exp", lambda x: x.exp(), math.exp, 0.4),
            ("silu", lambda x: x.silu(), numeric_silu, -0.7),
        ]

        for name, build_graph, numeric_function, x_value in cases:
            with self.subTest(name=name):
                x = Value(x_value)
                result = build_graph(x)
                result.backward()

                expected = central_difference(numeric_function, x_value)
                self.assertAlmostEqual(x.grad, expected, places=5)


class TestTinyNeuronTraining(unittest.TestCase):
    def test_linear_neuron_learns_known_function(self):
        # Training data follows y = 2x + 1 exactly.
        inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
        targets = [-3.0, -1.0, 1.0, 3.0, 5.0]

        weight = Value(-0.5)
        bias = Value(0.0)
        learning_rate = 0.05

        def mean_squared_error():
            predictions = [weight * x + bias for x in inputs]
            losses = [
                (prediction - target) ** 2
                for prediction, target in zip(predictions, targets)
            ]
            return sum(losses) / len(losses)

        initial_loss = mean_squared_error().value

        for _ in range(200):
            weight.grad = 0.0
            bias.grad = 0.0

            loss = mean_squared_error()
            loss.backward()

            weight.value -= learning_rate * weight.grad
            bias.value -= learning_rate * bias.grad

        final_loss = mean_squared_error().value

        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 1e-8)
        self.assertAlmostEqual(weight.value, 2.0, places=4)
        self.assertAlmostEqual(bias.value, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
