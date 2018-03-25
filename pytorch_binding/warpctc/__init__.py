# WarpCTC PyTorch bindings (c) 2018 by Thomas Viehmann <tv@lernapparat.de>
# All rights reserved. Licensed under the
# Apache License,  Version 2.0, January 2004
# see LICENSE in root directory

import torch

from . import _warpctc


class CTCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations, input_lengths, labels, label_lengths,
                reduce=True, size_average=False, length_average=False,
                blank_label=0, want_gradient=True):
        want_gradient &= activations.requires_grad
        # it would be cool to check torch.is_grad_enabled() here,
        # but that is always false due to how autograd works, thus
        # the little hack of passing in want_gradient
        costs, gradients = _warpctc.ctc(activations, input_lengths, labels,
                                        label_lengths, blank_label,
                                        want_gradient)
        if length_average:
            # we use label_lengths to not deal with blanks
            costs = costs / label_lengths
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


def ctc_loss(activations, input_lengths, labels, label_lengths,
             reduce=True, size_average=True, length_average=False,
             blank_label=0):
    r"""The Connecttionist Temopral Classification Loss

    Args:
        activations: FloatTensor of size `(TA, N, C)` where `TA` is the
            sequence length, `N` is the batch size, `C` is the number of
            classes (including blank). Can be GPU or CPU.
            These are the activations (log probabilities) for each
            sequence item.
        input_lengths: IntTensor of size `(N)` specifying the length of each
            `activation` in the minibatch. Must be on CPU.
        labels: IntTensor of size `(X)` where `X` is the sum of
            `label_lengths`. Must be on CPU.
        label_lengths: IntTensor of size `(N)` specifying the length of each
            label sequence.
        size_average (bool, optional):
            By default, the losses are averaged over observations for
            each minibatch. However, if the field :attr:`size_average` is set
            to ``False``,
            the losses are instead summed for each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged
            over observations for each minibatch, or summed, depending on
            :attr:`size_average`. When reduce is ``False``, returns a loss per
            batch instead and ignores :attr:`size_average`. Default: ``True``
        length_average (bool, optional):
            By default, the losses are summed over the sequence.
            However, if the field :attr:`length_average` is set to ``True``,
            the losses are normalized by the :attr:`label_lengths`.
            Default: ``False``
        blank_label (int, optional): Index of CTC blank label in the
            predictions. Default: 0
    """
    return CTCFunction.apply(activations, input_lengths, labels, label_lengths,
                             reduce, size_average, length_average, blank_label,
                             torch.is_grad_enabled())


class CTCLoss(torch.nn.Module):
    def __init__(self, reduce=True, size_average=True, length_average=False,
                 blank_label=0):
        super().__init__()
        self.reduce = reduce
        self.size_average = size_average
        self.length_average = length_average
        self.blank_label = blank_label

    def forward(self, activations, input_lengths, labels, label_lengths):
        return CTCFunction.apply(activations, input_lengths, labels,
                                 label_lengths, self.reduce, self.size_average,
                                 self.length_average, self.blank_label,
                                 torch.is_grad_enabled())
