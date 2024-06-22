"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.bias = None 
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), device = device, dtype = dtype)

        if bias: 
            self.bias = Parameter(init.kaiming_uniform(out_features, 1).transpose(), device = device, dtype = dtype)
        else: 
            self.bias = Parameter(init.zeros(1, out_features), device = device, dtype = dtype)

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.matmul(self.weight) + self.bias.broadcast_to((X.shape[0], self.out_features))
    
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        
        while len(X.shape) > 2:
            new_shape = X.shape  
            new_shape = new_shape[:-2] + (new_shape[-2]*new_shape[-1],)
            X = ops.reshape(X, new_shape)
        ### END YOUR SOLUTION
        return X


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules: 
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):

        y_one_hot = init.one_hot(logits.shape[-1], y)



        z_y = ops.summation(ops.multiply(logits, y_one_hot), axes=(-1,))

        ### BEGIN YOUR SOLUTION
        return ops.divide_scalar(ops.summation(ops.logsumexp(logits, axes = (-1,)) - z_y), float(1.0*logits.shape[0]))
    

        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim), device = device, dtype = dtype)
        self.bias = Parameter(init.zeros(dim), device = device, dtype = dtype )
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        c = 1
        for i in range(len(x.shape) - 1):
            c *= x.shape[i]
        ### BEGIN YOUR SOLUTION
        n = x.shape[-2]
        d = x.shape[-1]
        if self.training:
            

            mean = 1/x.shape[-2]*ops.summation(x, axes = (-2,))
            pre_broadcast_shape = x.shape[:-2] + (1, d)

            mean = mean.reshape(pre_broadcast_shape).broadcast_to(x.shape)
            var = 1/x.shape[-2]*ops.summation(ops.power_scalar(x - mean, 2), axes = (-2,))
            var = var.reshape(pre_broadcast_shape).broadcast_to(x.shape)
            

            # send (d,) to (1, ..., 1, d)
            # Then, send (1, ..., 1, d) to (*batch_size, d)


            running_mean = self.running_mean.reshape(pre_broadcast_shape).broadcast_to(x.shape)
            running_var = self.running_var.reshape(pre_broadcast_shape).broadcast_to(x.shape)

            running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            # bring back down to dimension of dim 
            self.running_mean = ops.summation(running_mean, axes = tuple(range(len(x.shape) - 1))).data/c
            self.running_var = ops.summation(running_var, axes = tuple(range(len(x.shape) - 1))).data/c
            y = (x - mean) / ops.power_scalar(var + self.eps, 0.5) * self.weight.reshape((1, self.dim)).broadcast_to(x.shape) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        else: 
            pre_broadcast_shape = x.shape[:-2] + (1, d)
            running_mean = self.running_mean.reshape(pre_broadcast_shape).broadcast_to(x.shape)
            running_var = self.running_var.reshape(pre_broadcast_shape).broadcast_to(x.shape)

            y = (x - running_mean) / ops.power_scalar(running_var + self.eps, 0.5) * self.weight.reshape((1, self.dim)).broadcast_to(x.shape) + self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
            # Use the running_mean/var directly 



        return y 
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        
        Ex = 1/x.shape[-1]*ops.summation(x, axes = (-1,))
        

        Ex = ops.broadcast_to(ops.reshape(Ex, Ex.shape + (1,)), x.shape)
        #print(Ex, Ex.shape)
        # x is of size (B, F), then Ex will be of size (B, 1)

        varx = 1/x.shape[-1]*ops.summation(ops.power_scalar(x - Ex, 2), axes = (-1,)) 
        varx = ops.broadcast_to(varx.reshape(varx.shape + (1,)), x.shape)

        #print(Ex.shape,varx.shape)
        
        x = (x - Ex) / ops.power_scalar(varx + self.eps, 0.5) * self.weight.reshape((1,self.dim)).broadcast_to(x.shape)  + self.bias.reshape((1,self.dim)).broadcast_to(x.shape)
        
        return x
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training: 
            drop_mask = init.randb(*x.shape, p=1 - self.p)
            
            return 1/(1-self.p)*x * drop_mask 


        return x 
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
