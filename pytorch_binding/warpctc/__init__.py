import torch

from . import _warpctc

#std::tuple<at::Tensor, at::Tensor> ctc(at::Tensor activations,
#					   at::Tensor input_lengths,
#					   at::Tensor labels,
#					   at::Tensor label_lengths, int blank_label, bool want_gradient)

class CTCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations, input_lengths, labels, label_lengths,
                reduce=True, size_average=False, length_average=False, blank_label = 0, want_gradient = True):
        want_gradient &= activations.requires_grad # it would be cool to check torch.is_grad_enabled() here, 
                                                   # but that is always false due to how autograd works, thus
                                                   # the little hack above
        costs, gradients = _warpctc.ctc(activations, input_lengths, labels, label_lengths, blank_label, want_gradient)
        if length_average:
            costs = costs / label_lengths # we use label_lengths to not deal with blanks
            if want_gradient:
                if gradients.is_cuda:
                    label_lengths = label_lengths.cuda(device=gradients.device)
                gradients /= label_lengths.view(1, -1, 1)
        if reduce:
            if size_average:
                costs = costs.mean()
            else:
                costs = costs.sum()
            if want_gradient and size_average:
                gradients /= label_lengths.size(0)
        ctx.save_for_backward(gradients)
        return costs

    @staticmethod
    def backward(ctx, grad_out):
        # check whether grad_in is zero
        gradients, = ctx.saved_variables
        if gradients.is_cuda:
            grad_out = grad_out.cuda(device=gradients.get_device())
        gradients = gradients * grad_out.view(1, -1, 1)
        return gradients, None, None, None, None, None, None, None, None


class CTCLoss(torch.nn.Module):
    def __init__(self, reduce=True, size_average=True, length_average=False, blank_label = 0):
        super().__init__()
        self.reduce = reduce
        self.size_average = size_average
        self.length_average = length_average
        self.blank_label = blank_label
    def forward(self, activations, input_lengths, labels, label_lengths):
        return CTCFunction.apply(activations, input_lengths, labels, label_lengths,
                                 self.reduce, self.size_average, self.length_average, self.blank_label,
                                 torch.is_grad_enabled())