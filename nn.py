import random

from minigrad import Value


class Module:
    # list of values that we will update
    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

# ok basically just something that looks like
# Neuron([x1, x2, x3]) = (w1 * x1 + w2 * x2 + w3 * x3) + bias
class Neuron(Module):
    def __init__(self, size: int, nonlinear: bool = True):
        # weights can be negative
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(size)]
        # default biases to 0
        self.bias = Value(0.0)
        self.nonlinear = nonlinear

    # we take in the [x1, x2, x3]
    def __call__(self, inputs) -> Value:
        activation = self.bias

        # neater way, traverse weight and input at same time
        for w, i in zip(self.weights, inputs):
            activation += w * i

        if self.nonlinear:
            return activation.silu()

        return activation

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]


# u just take a bunch of neurons, throw the same input to all of em, and return their outputs
class Layer(Module):
    def __init__(self, input_size: int, output_size: int, nonlinear: bool = True):
        # so the size of each neuron is input_size
        # and the amount of neurons is output size
        self.neurons = [Neuron(input_size, nonlinear) for _ in range(output_size)]

    # we take in the [x1, x2, x3]
    def __call__(self, inputs) -> list[Value]:
        return [n(inputs) for n in self.neurons]


    def parameters(self) -> list[Value]:
        total_params = []
        for n in self.neurons:
            total_params += n.parameters()
        return total_params

class MLP(Module):
    def __init__(self, input_size: int, layer_sizes) -> None:
        # so the input size is again the size of each of the neuron
        # then based on layer sizes, thats how many outputs comes out of each layer, which is the amount of neurons
        #
        # for example: input_size = 2, layer_sizes = [3, 4, 1]
        # [x1, x2] -> layer 3 -> [y1, y2, y3] -> layer 4 -> [z1, z2, z3, z4] -> layer 1 -> final answer

        self.layers = []
        currInputSize = input_size
        for outputSize in layer_sizes[:-1]:
            self.layers.append(Layer(currInputSize, outputSize))
            currInputSize = outputSize

        self.layers.append(Layer(currInputSize, layer_sizes[-1], False))


    # we take in the [x1, x2, x3]
    def __call__(self, inputs) -> list[Value]:
        currOutputs = [value for value in inputs]
        for layer in self.layers:
            currOutputs = layer(currOutputs)

        return currOutputs


    def parameters(self) -> list[Value]:
        total_params = []
        for l in self.layers:
            total_params += l.parameters()
        return total_params
