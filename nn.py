from ast import Mod
import enum

from minigrad import Value
import random


class Module:
    # list of values that we will update
    def parameters(self)-> list[Value]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

# ok basically just something that looks like
# Neuron([x1, x2, x3]) = (w1 * x1 + w2 * x2 + w3 * x3) + bias
class Neuron(Module):
    def __init__(self, size, nonlinear=True):
        # weights can be negative
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(size)]
        # default biases to 0
        self.bias = Value(0.0)
        self.nonlinear = nonlinear

    # we take in the [x1, x2, x3]
    def __call__(self, input: list[Value]):
        activation = self.bias

        # neater way, traverse weight and input at same time
        for w, i in zip(self.weights, input):
            activation += w * i

        if self.nonlinear:
            return activation.silu()

        return activation

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]



# TODO: Create a Layer containing a chosen number of Neuron objects.
# TODO: Implement Layer.__call__() by passing the same inputs into every Neuron.
# TODO: Return one Value for a one-neuron layer and a list for multiple neurons.
# TODO: Make Layer.parameters() flatten and return every Neuron parameter.

# TODO: Create an MLP from an input size and a list of output sizes per layer.
# TODO: Make hidden MLP layers nonlinear and the final layer linear.
# TODO: Implement MLP.__call__() by feeding each layer's output into the next.
# TODO: Make MLP.parameters() flatten and return every Layer parameter.

# TODO: Test the parameter count of a small known architecture by hand.
# TODO: Test Neuron, Layer, and MLP forward-output shapes.
# TODO: Test that every MLP parameter receives a finite gradient.

# TODO: Create a tiny nonlinear XOR-style training example with fixed seeds.
# TODO: Compute a scalar loss from all predictions and targets.
# TODO: Zero gradients, call loss.backward(), and update every parameter with SGD.
# TODO: Verify that loss decreases and the final predictions classify XOR correctly.
