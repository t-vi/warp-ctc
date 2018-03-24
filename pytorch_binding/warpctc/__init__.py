import math
from torch import nn
from torch.autograd import Function
import torch

from . import _warpctc

#std::tuple<at::Tensor, at::Tensor> ctc(at::Tensor activations,
#					   at::Tensor input_lengths,
#					   at::Tensor labels,
#					   at::Tensor label_lengths, int blank_label, bool want_gradient)

class CTCFunction(Function):
    @staticmethod
    def forward(ctx, activations, input_lengths, labels, label_lengths, blank_label = 0):
        want_gradient = activations.requires_grad # it would be cool to check torch.is_grad_enabled() here, 
                                                  # but that is always false due to how autograd works
        costs, gradients = _warpctc.ctc(activations, input_lengths, labels, label_lengths, blank_label, want_gradient)
        ctx.save_for_backward(gradients)
        return costs

    @staticmethod
    def backward(ctx, grad_out):
        # check whether grad_in is zero
        gradients, = ctx.saved_variables
        if gradients.is_cuda:
            grad_out = grad_out.cuda(device=gradients.get_device())
        gradients = gradients * grad_out.view(1, -1, 1)
        return gradients, None, None, None, None


