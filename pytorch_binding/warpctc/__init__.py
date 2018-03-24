import math
from torch import nn
from torch.autograd import Function
import torch

from . import _warpctc

#std::tuple<at::Tensor, at::Tensor> ctc(at::Tensor activations,
#					   at::Tensor input_lengths,
#					   at::Tensor labels,
#					   at::Tensor label_lengths, int blank_label)

class CTCFunction(Function):
    @staticmethod
    def forward(ctx, activations, input_lengths, labels, label_lengths, blank_label = 0):
        costs, gradients = _warpctc.ctc(activations, input_lengths, labels, label_lengths, blank_label)
        ctx._act_dim = activations.dim()
        ctx._input_lengths = input_lengths
        ctx.save_for_backward(gradients)
        return costs

    @staticmethod
    def backward(ctx, grad_out):
        # check whether grad_in is zero
        gradients, = ctx.saved_variables
        gradients = gradients * grad_out.view(1, -1, 1)
        return gradients, None, None, None, None


