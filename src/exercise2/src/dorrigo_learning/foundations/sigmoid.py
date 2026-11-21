import math
from .autograd import Value
def exp(self) -> Value:
    # TODO:
    # 1. Calculate e^self.data
    # 2. Create the output Value
    # 3. Connect it to the computation graph
    # 4. Implement its backward derivative
    output = Value(math.exp(self.data), _children=(self,))
    output.backward()
    return output

def sigmoid(self) -> Value:
    # TODO:
    # Implement sigmoid using Value operations.
    # Preserve the computation graph.
    val = exp(self)
    output = 1 / (1 + val.data)
    #output._backward = lambda: self.grad += output.data * (1 - output.data)
    return Value(output, _children=(self,))

