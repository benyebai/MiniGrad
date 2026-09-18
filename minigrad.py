import math

class Value:
    def __init__(self, value, children=(), name=""):
        self.value = value
        self.children = set(children)
        self.name = name
        self.grad = 0
        self._backward = lambda: None

    def __add__(self, other):
        # python is stupid so we use it to do scalars
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.value + other.value, (self, other))

        def _backward():
            # since its addition, my grad is just
            # the deriv of my "parent" * the deriv of myself to my parent which is 1 in the addition case
            self.grad += 1 * res.grad
            other.grad += 1 * res.grad

        # then we install this backward function to our result! So when our result calls backwards he knows how to update us
        res._backward = _backward

        return res

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.value * other.value, (self, other))

        def _backward():
            self.grad += other.value * res.grad
            other.grad += self.value * res.grad

        # then we install this backward function to our result! So when our result calls backwards he knows how to update us
        res._backward = _backward

        return res

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        # removed the ability for the exponent to be a value, because it screws shi up
        res = Value(self.value ** exponent, (self, ))

        def _backward():
            # ok so it would look like
            # x ** y = z
            # then dz/dy = (x ** y) * lnx
            self.grad += (exponent * (self.value ** (exponent - 1))) * res.grad

        res._backward = _backward

        return res

    # this one a bit tricky, but following the same concept of creating a new value made it clearer to me
    def __neg__(self):

        res = Value(-self.value, (self,))

        def _backward():
            # ok so it would look like
            # y = -x
            # then dy/dx = -1
            self.grad += -1 * res.grad

        res._backward = _backward

        return res

    def exp(self):
        res = Value(math.exp(self.value), (self,))

        def _backward():
            # ok so it would look like
            # y = e^x, uh if i remember correct
            # then dy/dx = e^x
            self.grad += math.exp(self.value) * res.grad

        res._backward = _backward

        return res

    # now instead of the complicated code just combine add and negate ezpz
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        # and the reason why we warp it in a value is just because
        # we know always in this case other is a number not a var
        return Value(other) - self

    # right same thing we can use the previous defintions to create our divison
    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return Value(other) / self

    # i see since u already have the existing pieces keep using what u already created, u dont need to do all the backprop
    # anymore, u created all the building blocks.
    def silu(self):
        sigmoid = Value(1) / (Value(1) + (-self).exp())
        return self * sigmoid


    def backward(self):
        topologialReverseOrder = []
        visited = set()

        # its really not that deep, just make sure u traverse the graph once and add to list after u visit all its children
        def create_top_reverse_order(node):
            if node not in visited:
                visited.add(node)

                for n in node.children:
                    create_top_reverse_order(n)

                topologialReverseOrder.append(node)

        create_top_reverse_order(self)

        # this is important haha oops
        self.grad = 1.0

        for n in reversed(topologialReverseOrder):
            n._backward()
