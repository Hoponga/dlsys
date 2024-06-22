"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
        for param in params: 
            self.u[hash(param)] = 0


    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None: 
                continue 
            param_grad = ndl.Tensor(param.grad.data, dtype = param.data.dtype) 

            hashed = hash(param)
            self.u[hashed] = self.momentum*self.u[hashed] + (1 - self.momentum)*(param_grad.data + self.weight_decay * param.data)
            #print(param_grad.data.dtype)
            # why is the grad of dtype float64 but the normal data is of dtype float32

            #print(self.u[hashed].dtype)
            param.data = param.data - self.lr*self.u[hashed]


             

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

        for param in params: 
            self.m[hash(param)] = 0
            self.v[hash(param)] = 0
        
        

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params: 
            if param.grad is None: 
                continue
            param_grad = ndl.Tensor(param.grad.data + self.weight_decay*param.data, dtype = param.data.dtype)
            key = hash(param)
            new_m_bias = self.beta1*self.m[key] + (1 - self.beta1)*param_grad
            new_v_bias = self.beta2*self.v[key] + (1 - self.beta2)*param_grad*param_grad
            self.m[key] = new_m_bias 
            self.v[key] = new_v_bias
            m = new_m_bias/(1-self.beta1**self.t)
            v = new_v_bias/(1-self.beta2**self.t)

            param.data = param.data - self.lr*(m/(ndl.ops.power_scalar(v, 0.5) + self.eps))
        
        ### END YOUR SOLUTION
