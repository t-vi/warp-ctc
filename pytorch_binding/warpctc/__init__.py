import math
from torch import nn
from torch.autograd import Function
import torch

import _warpctc

torch.manual_seed(42)


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


class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
