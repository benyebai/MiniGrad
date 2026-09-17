from minigrad import Value
import random

# TODO: Create a Module base class with parameters() returning an empty list.
# TODO: Add Module.zero_grad() to reset every parameter gradient to zero.

class Module:
    # list of values that we will update
    def parameters(self)-> list[Value]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad



# TODO: Create a Neuron initialized with a chosen number of inputs.
# TODO: Give each Neuron one random Value weight per input and one Value bias.
# TODO: Implement Neuron.__call__() as sum(weight * input) + bias.
# TODO: Apply SiLU when the Neuron is nonlinear and return the raw sum otherwise.
# TODO: Make Neuron.parameters() return all weights followed by the bias.

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
