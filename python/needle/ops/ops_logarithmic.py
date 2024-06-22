from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        return logsumexp(Z) - array_api.max(Z)

    def gradient(self, out_grad, node):
        return 2


def logsoftmax(a):
    return LogSoftmax()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        Z_max = array_api.max(Z, axis = self.axes, keepdims = True)
        
        out = array_api.log(array_api.sum(array_api.exp(Z - Z_max), axis = self.axes, keepdims = True)) + Z_max
        new_shape = [] 
        new_shape = []
        if self.axes:
            l = len(Z.shape)
            for i, n in enumerate(Z.shape):
                if (i not in self.axes) and ((i - l) not in self.axes):
                    new_shape.append(n)
            out = out.reshape(new_shape).astype(Z.dtype)
        else:
            # for test
            out = float(out)

        return out
        



        
    # Derivative of log sum exp is 1/sum * e^z_j
    # Vectorizing, we get that 
    def gradient(self, out_grad, node):
        a = node.inputs[0]
        a_max = array_api.max(a.numpy(), axis = self.axes, keepdims = True)
        exp_z = array_api.exp(a.numpy() - a_max)
        exp_sum = array_api.sum(exp_z, axis = self.axes, keepdims = True)
        
        grad_result = exp_z/exp_sum 
        # Then, we need to broadcast it back to the shape of a 

        new_shape = list(a.numpy().shape)

        # Suppose we have grad of the shape (a, b, c)
        # For 
        if self.axes: 
            for i in self.axes: 
                new_shape[i] = 1
            grad = reshape(out_grad, new_shape) 

        else: 
            grad = out_grad 

        #print(grad.shape, grad_result.shape, a.shape)
        return broadcast_to(grad, a.shape) * Tensor(grad_result, dtype = grad.dtype)




def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
    

