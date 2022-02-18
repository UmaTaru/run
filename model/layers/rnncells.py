




# # c








# import threading
# import torch.cuda
# from .utils import THNN_H_PATH, THCUNN_H_PATH, parse_header, load_backend
# # import THNN_H_PATH, THCUNN_H_PATH, parse_header, load_backend


# class Backends(object):

#     def __init__(self):
#         self.backends = {}

#     def __getattr__(self, name):
#         return self.backends[name].load()

#     def __getitem__(self, name):
#         return self.backends[name].load()


# class Backend(object):

#     def __init__(self, lib_prefix, lib_name, functions, mixins=tuple()):
#         self.lib_prefix = lib_prefix
#         self.lib_name = lib_name
#         self.functions = functions
#         self.mixins = mixins
#         self.backend = None
#         self.loading_lock = threading.Lock()

#     def load(self):
#         # This looks a little weird, but it's neccesary for thread safe loading.
#         # Loading the backend can take some time, so multiple threads can enter
#         # the if clause. We have to ensure that only the first one to acquire
#         # the lock will actually load the backend, and that the rest won't
#         # do it again.
#         if self.backend is None:
#             with self.loading_lock:
#                 if self.backend is None:
#                     self.backend = load_backend(self.lib_prefix, self.lib_name,
#                                                 self.functions, self.mixins)
#         return self.backend


# class THNNCudaBackendStateMixin(object):

#     @property
#     def library_state(self):
#         return torch.cuda._state_cdata


# type2backend = Backends()

# _thnn_headers = parse_header(THNN_H_PATH)
# _thcunn_headers = parse_header(THCUNN_H_PATH)

# for t in ['Float', 'Double']:
#     backend = Backend(t, 'torch._thnn._THNN', _thnn_headers)

#     type2backend.backends['THNN{}Backend'.format(t)] = backend
#     type2backend.backends['torch.{}Tensor'.format(t)] = backend
#     type2backend.backends[getattr(torch, '{}Tensor'.format(t))] = backend


# for t in ['Half', '', 'Double']:
#     backend = Backend('Cuda' + t, 'torch._thnn._THCUNN', _thcunn_headers, (THNNCudaBackendStateMixin,))
#     type2backend.backends['THNNCuda{}Backend'.format(t)] = backend
#     py_name = 'Float' if t == '' else t
#     type2backend.backends['torch.cuda.{}Tensor'.format(py_name)] = backend
#     type2backend.backends[getattr(torch.cuda, '{}Tensor'.format(py_name))] = backend
# import threading
# import torch.cuda
# from utils import THNN_H_PATH, THCUNN_H_PATH, parse_header, load_backend


# class Backends(object):

#     def __init__(self):
#         self.backends = {}

#     def __getattr__(self, name):
#         return self.backends[name].load()

#     def __getitem__(self, name):
#         return self.backends[name].load()


# class Backend(object):

#     def __init__(self, lib_prefix, lib_name, functions, mixins=tuple()):
#         self.lib_prefix = lib_prefix
#         self.lib_name = lib_name
#         self.functions = functions
#         self.mixins = mixins
#         self.backend = None
#         self.loading_lock = threading.Lock()

#     def load(self):
#         # This looks a little weird, but it's neccesary for thread safe loading.
#         # Loading the backend can take some time, so multiple threads can enter
#         # the if clause. We have to ensure that only the first one to acquire
#         # the lock will actually load the backend, and that the rest won't
#         # do it again.
#         if self.backend is None:
#             with self.loading_lock:
#                 if self.backend is None:
#                     self.backend = load_backend(self.lib_prefix, self.lib_name,
#                                                 self.functions, self.mixins)
#         return self.backend


# class THNNCudaBackendStateMixin(object):

#     @property
#     def library_state(self):
#         return torch.cuda._state_cdata


# type2backend = Backends()

# _thnn_headers = parse_header(THNN_H_PATH)
# _thcunn_headers = parse_header(THCUNN_H_PATH)

# for t in ['Float', 'Double']:
#     backend = Backend(t, 'torch._thnn._THNN', _thnn_headers)

#     type2backend.backends['THNN{}Backend'.format(t)] = backend
#     type2backend.backends['torch.{}Tensor'.format(t)] = backend
#     type2backend.backends[getattr(torch, '{}Tensor'.format(t))] = backend


# for t in ['Half', '', 'Double']:
#     backend = Backend('Cuda' + t, 'torch._thnn._THCUNN', _thcunn_headers, (THNNCudaBackendStateMixin,))
#     type2backend.backends['THNNCuda{}Backend'.format(t)] = backend
#     py_name = 'Float' if t == '' else t
#     type2backend.backends['torch.cuda.{}Tensor'.format(py_name)] = backend
#     type2backend.backends[getattr(torch.cuda, '{}Tensor'.format(py_name))] = backend



# import os
# import itertools
# import importlib

# THNN_H_PATH = os.path.join(os.path.dirname(__file__), '..', 'lib', 'THNN.h')
# THCUNN_H_PATH = os.path.join(os.path.dirname(__file__), '..', 'lib', 'THCUNN.h')


# def _unpickle_backend(backend_name):
#     import torch._thnn
#     return torch._thnn.type2backend[backend_name]


# class THNNBackendBase(object):

#     def __init__(self):
#         self.methods = {}

#     def __getattr__(self, name):
#         method = self.methods.get(name, None)
#         if method is None:
#             raise NotImplementedError
#         return method

#     def register_method(self, name, ctypes_fn):
#         self.methods[name] = ctypes_fn

#     @property
#     def library_state(self):
#         return 0

#     def __reduce__(self):
#         return (_unpickle_backend, (type(self).__name__,))


# class Function(object):

#     def __init__(self, name):
#         self.name = name
#         self.arguments = []

#     def add_argument(self, arg):
#         assert isinstance(arg, Argument)
#         self.arguments.append(arg)

#     def __repr__(self):
#         return self.name + '(' + ', '.join(map(lambda a: a.__repr__(), self.arguments)) + ')'


# class Argument(object):

#     def __init__(self, _type, name, is_optional):
#         self.type = _type
#         self.name = name
#         self.is_optional = is_optional

#     def __repr__(self):
#         return self.type + ' ' + self.name


# def parse_header(path):
#     with open(path, 'r') as f:
#         lines = f.read().split('\n')

#     # Remove empty lines and preprocessor directives
#     lines = filter(lambda l: l and not l.startswith('#'), lines)
#     # Remove line comments
#     lines = map(lambda l: l.partition('//'), lines)
#     # Select line and comment part
#     lines = map(lambda l: (l[0].strip(), l[2].strip()), lines)
#     # Remove trailing special signs
#     lines = map(lambda l: (l[0].rstrip(');').rstrip(','), l[1]), lines)
#     # Split arguments
#     lines = map(lambda l: (l[0].split(','), l[1]), lines)
#     # Flatten lines
#     new_lines = []
#     for l, c in lines:
#         for split in l:
#             new_lines.append((split, c))
#     lines = new_lines
#     del new_lines
#     # Remove unnecessary whitespace
#     lines = map(lambda l: (l[0].strip(), l[1]), lines)
#     # Remove empty lines
#     lines = filter(lambda l: l[0], lines)
#     generic_functions = []
#     for l, c in lines:
#         if l.startswith('TH_API void THNN_'):
#             fn_name = l.lstrip('TH_API void THNN_')
#             if fn_name[0] == '(' and fn_name[-2] == ')':
#                 fn_name = fn_name[1:-2]
#             else:
#                 fn_name = fn_name[:-1]
#             generic_functions.append(Function(fn_name))
#         elif l:
#             t, name = l.split()
#             if '*' in name:
#                 t = t + '*'
#                 name = name[1:]
#             generic_functions[-1].add_argument(Argument(t, name, '[OPTIONAL]' in c))
#     return generic_functions


# def load_backend(t, lib, generic_functions, mixins=tuple()):
#     lib_handle = importlib.import_module(lib)
#     backend_name = 'THNN{}Backend'.format(t)
#     backend = type(backend_name, mixins + (THNNBackendBase,), {})()
#     for function in generic_functions:
#         full_fn_name = '{}{}'.format(t, function.name)
#         fn = getattr(lib_handle, full_fn_name)
#         backend.register_method(function.name, fn)
#     return backend
# import os
# import itertools
# import importlib

# THNN_H_PATH = os.path.join(os.path.dirname(__file__), '..', 'lib', 'THNN.h')
# THCUNN_H_PATH = os.path.join(os.path.dirname(__file__), '..', 'lib', 'THCUNN.h')


# def _unpickle_backend(backend_name):
#     import torch._thnn
#     return torch._thnn.type2backend[backend_name]


# class THNNBackendBase(object):

#     def __init__(self):
#         self.methods = {}

#     def __getattr__(self, name):
#         method = self.methods.get(name, None)
#         if method is None:
#             raise NotImplementedError
#         return method

#     def register_method(self, name, ctypes_fn):
#         self.methods[name] = ctypes_fn

#     @property
#     def library_state(self):
#         return 0

#     def __reduce__(self):
#         return (_unpickle_backend, (type(self).__name__,))


# class Function(object):

#     def __init__(self, name):
#         self.name = name
#         self.arguments = []

#     def add_argument(self, arg):
#         assert isinstance(arg, Argument)
#         self.arguments.append(arg)

#     def __repr__(self):
#         return self.name + '(' + ', '.join(map(lambda a: a.__repr__(), self.arguments)) + ')'


# class Argument(object):

#     def __init__(self, _type, name, is_optional):
#         self.type = _type
#         self.name = name
#         self.is_optional = is_optional

#     def __repr__(self):
#         return self.type + ' ' + self.name


# def parse_header(path):
#     with open(path, 'r') as f:
#         lines = f.read().split('\n')

#     # Remove empty lines and preprocessor directives
#     lines = filter(lambda l: l and not l.startswith('#'), lines)
#     # Remove line comments
#     lines = map(lambda l: l.partition('//'), lines)
#     # Select line and comment part
#     lines = map(lambda l: (l[0].strip(), l[2].strip()), lines)
#     # Remove trailing special signs
#     lines = map(lambda l: (l[0].rstrip(');').rstrip(','), l[1]), lines)
#     # Split arguments
#     lines = map(lambda l: (l[0].split(','), l[1]), lines)
#     # Flatten lines
#     new_lines = []
#     for l, c in lines:
#         for split in l:
#             new_lines.append((split, c))
#     lines = new_lines
#     del new_lines
#     # Remove unnecessary whitespace
#     lines = map(lambda l: (l[0].strip(), l[1]), lines)
#     # Remove empty lines
#     lines = filter(lambda l: l[0], lines)
#     generic_functions = []
#     for l, c in lines:
#         if l.startswith('TH_API void THNN_'):
#             fn_name = l.lstrip('TH_API void THNN_')
#             if fn_name[0] == '(' and fn_name[-2] == ')':
#                 fn_name = fn_name[1:-2]
#             else:
#                 fn_name = fn_name[:-1]
#             generic_functions.append(Function(fn_name))
#         elif l:
#             t, name = l.split()
#             if '*' in name:
#                 t = t + '*'
#                 name = name[1:]
#             generic_functions[-1].add_argument(Argument(t, name, '[OPTIONAL]' in c))
#     return generic_functions


# def load_backend(t, lib, generic_functions, mixins=tuple()):
#     lib_handle = importlib.import_module(lib)
#     backend_name = 'THNN{}Backend'.format(t)
#     backend = type(backend_name, mixins + (THNNBackendBase,), {})()
#     for function in generic_functions:
#         full_fn_name = '{}{}'.format(t, function.name)
#         fn = getattr(lib_handle, full_fn_name)
#         backend.register_method(function.name, fn)
#     return backend






















# Modified from OpenNMT.py, Z-forcing
import torch
from torch.autograd.function import Function, InplaceFunction
#from torch._thnn import type2backend
#import torch._thnn


class GRUFused(Function):
    def __init__(self):
        self.backend = None

    def forward(self, input_gate, hidden_gate, hx, ibias=None, hbias=None):
        if self.backend is None:
            self.backend = type2backend[type(input_gate)]
        hy = input_gate.new()
        if ibias is not None:
            if ibias.dim() == 1:
                ibias.unsqueeze_(0)
            if hbias.dim() == 1:
                hbias.unsqueeze_(0)

        self.backend.GRUFused_updateOutput(
            self.backend.library_state,
            input_gate, hidden_gate, ibias, hbias, hx, hy)
        self.save_for_backward(input_gate, hidden_gate, ibias)
        return hy

    def backward(self, gradOutput):
        if self.backend is None:
            self.backend = type2backend[type(grad_output)]
        gradInput = gradOutput.new()
        input_gate, hidden_gate, bias = self.saved_tensors

        self.backend.GRUFused_updateGradInput(
            self.backend.library_state,
            input_gate, hidden_gate, gradOutput, gradInput)
        if bias is not None:
            gb1 = input_gate.sum(0).squeeze()
            gb2 = hidden_gate.sum(0).squeeze()

            return input_gate, hidden_gate, gradInput, gb1, gb2
        else:
            return input_gate, hidden_gate, gradInput


class LSTMFused(Function):
    def __init__(self):
        self.backend = None

    def forward(self, input_gate, hidden_gate, cx, ibias=None, hbias=None):
        if self.backend is None:
            self.backend = type2backend[type(input_gate)]
        hy = input_gate.new()
        cy = input_gate.new()
        if ibias is not None:
            if ibias.dim() == 1:
                ibias.unsqueeze_(0)
            if hbias.dim() == 1:
                hbias.unsqueeze_(0)
        self.backend.LSTMFused_updateOutput(
            self.backend.library_state,
            input_gate, hidden_gate,
            ibias, hbias,
            cx, hy, cy)
        self.save_for_backward(input_gate, hidden_gate, cx, cy, ibias)
        return hy, cy

    def backward(self, *gradOutput):
        if self.backend is None:
            self.backend = type2backend[type(gradOutput[0])]

        gradInput = gradOutput[0].new()
        gradInputCell = gradOutput[0].new()
        saved_tens, local_go, cx, cy, bias = self.saved_tensors

        self.backend.LSTMFused_updateGradInput(
            self.backend.library_state,
            saved_tens, local_go, cx, cy,
            gradOutput[0], gradOutput[1], gradInput)

        if bias is not None:
            gb1 = local_go.sum(0).squeeze()
            gb2 = local_go.sum(0).squeeze()

            return local_go, local_go, gradInput, gb1, gb2
        else:
            return local_go, local_go, gradInput












import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn._functions.thnn.rnnFusedPointwise import LSTMFused, GRUFused

class StackedLSTMCell(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h_c):
        """
        Args:
            x: [batch_size, input_size]
            h_c: [2, num_layers, batch_size, hidden_size]
        Return:
            last_h_c: [2, batch_size, hidden_size] (h from last layer)
            h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            # h of i-th layer
            h_i, c_i = layer(x, (h_0[i], c_0[i]))

            # x for next layer
            x = h_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_list += [h_i]
            c_list += [c_i]

        last_h_c = (h_list[-1], c_list[-1])
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        h_c_list = (h_list, c_list)

        return last_h_c, h_c_list


class StackedGRUCell(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRUCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h):
        """
        Args:
            x: [batch_size, input_size]
            h: [num_layers, batch_size, hidden_size]
        Return:
            last_h: [batch_size, hidden_size] (h from last layer)
            h_list: [num_layers, batch_size, hidden_size] (h from all layers)
        """
        # h of all layers
        h_list = []
        for i, layer in enumerate(self.layers):
            # h of i-th layer
            h_i = layer(x, h[i])

            # x for next layer
            x = h_i
            if i + 1 is not self.num_layers:
                x = self.dropout(x)
            h_list.append(h_i)

        last_h = h_list[-1]
        h_list = torch.stack(h_list)

        return last_h, h_list
