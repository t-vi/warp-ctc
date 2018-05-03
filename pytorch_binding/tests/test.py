#!/usr/bin/python3

import unittest

import torch
import warpctc

# tests from upstream warpctc test/test_cpu.cpp


class TestWarpCTC(unittest.TestCase):
    def small_test(self, cuda=False):
        # alphabet_size = 5
        seq_len = 2
        activations = torch.FloatTensor(
            [[0.1, 0.6, 0.1, 0.1, 0.1],
             [0.1, 0.1, 0.6, 0.1, 0.1]]).unsqueeze(1)
        probs = torch.nn.functional.softmax(activations, -1)
        if cuda:
            activations = activations.cuda()
            # Score calculation is specific to the given activations above
        expected_score = probs[0, 0, 1] * probs[1, 0, 2]

        labels = torch.IntTensor([1, 2])
        label_lengths = torch.IntTensor([2])
        lengths = torch.IntTensor([seq_len])

        score = warpctc.CTCFunction.apply(activations, lengths, labels,
                                          label_lengths)

        score = torch.exp(-score)
        eps = 1e-6
        self.assertTrue(expected_score-eps < score < expected_score + eps,
                        msg="CTCloss not in expected interval")

    def options_test(self, cuda):
        # alphabet_size = 6
        # seq_len = 5
        # minibatch = 2
        # timestep x batch x alphabet
        activations = torch.tensor([
          [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
           [0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508]],
          [[0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
           [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549]],
          [[0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
           [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456]],
          [[0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
           [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345]],
          [[0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107],
           [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]],
          requires_grad=True)

        expected_grads = torch.FloatTensor([  # from test_cpu from tensorflow
          [[-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
           [-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508]],
          [[0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436],
           [0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549]],
          [[0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688],
           [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544]],
          [[0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533],
           [0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345]],
          [[-0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107],
           [-0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]])

        # Calculate the expected scores analytically, well, one of them
        expected_scores = torch.FloatTensor([
                 -torch.log(activations[0, 0, 0] * activations[1, 0, 1]
                            * activations[2, 0, 2]
                            * activations[3, 0, 1] * activations[4, 0, 0]),
                 5.42262  # from tensorflow
                 ])

        if cuda:
            activations = activations.cuda()
            expected_grads = expected_grads.cuda()

        activations = torch.log(activations)
        activations.retain_grad()

        labels = torch.IntTensor([0, 1, 2, 1, 0,
                                  0, 1, 1, 0])

        label_lengths = torch.IntTensor([5, 4])
        lengths = torch.IntTensor([5, 5])

        ctcloss = warpctc.CTCLoss(reduce=False, blank_label=5)
        scores = ctcloss(activations, lengths, labels, label_lengths)

        scores.sum().backward()

        eps = 1e-4
        self.assertTrue((scores-expected_scores).abs().max() < eps,
                        msg="CTCloss not in expected interval")
        self.assertTrue((activations.grad-expected_grads).abs().max() < eps,
                        "CTCloss gradient not in expected interval")

    def inf_test(self, cuda=False):
        alphabet_size = 15
        seq_len = 50
        label_len = 10
        minibatch = 1

        labels = torch.IntTensor(label_len).random_(1, alphabet_size)
        if label_len >= 3:  # guarantee repeats for testing
            labels[label_len // 2] = labels[label_len // 2 + 1]
            labels[label_len // 2 - 1] = labels[label_len // 2]
        labels[0] = 2

        label_lengths = torch.IntTensor([label_len])

        acts = torch.rand(seq_len, minibatch, alphabet_size)
        acts[:, 0, 2] = -1e30

        if cuda:
            acts = acts.cuda()

        acts = torch.tensor(acts, requires_grad=True, dtype=acts.dtype)

        sizes = torch.IntTensor([seq_len])

        cost = warpctc.CTCFunction.apply(acts, sizes, labels, label_lengths)

        self.assertEqual(cost, float("inf"), msg="cost should be inf")

        cost.sum().backward()

        # chec for NaN g==g is true unless it is NaN
        self.assertTrue((acts.grad == acts.grad).all(),
                        msg="gradient of inf cost should not be NaN")

    def grad_test(self, cuda=False):
        device = torch.device("cuda:0" if cuda else "cpu")
        problem_sizes = [[20, 50, 15, 1, 10**(-2.5)],
                         [5, 10, 5, 65, 1e-2]]
        # n.b. warpctc's c++ cpu tests use squared relative difference

        for alphabet_size, seq_len, label_len, minibatch, tol in problem_sizes:
            acts = torch.rand(seq_len, minibatch, alphabet_size,
                              requires_grad=True,
                              dtype=torch.float32,
                              device=device)
            sizes = torch.IntTensor(minibatch).fill_(seq_len)
            label_lengths = torch.IntTensor(minibatch).fill_(label_len)
            labels = torch.IntTensor(label_len,
                                     minibatch).random_(1, alphabet_size)
            if label_len >= 3:  # guarantee repeats for testing
                labels[label_len // 2] = labels[label_len // 2 + 1]
                labels[label_len // 2 - 1] = labels[label_len // 2]
            labels = labels.view(-1)
            torch.autograd.gradcheck(warpctc.CTCFunction.apply,
                                     (acts, sizes, labels, label_lengths),
                                     rtol=tol, atol=tol, eps=1e-1)

    def test_small_cpu(self):
        self.small_test(cuda=False)

    def test_small_gpu(self):
        self.small_test(cuda=True)

    def test_options_cpu(self):
        self.options_test(cuda=False)

    def test_options_gpu(self):
        self.options_test(cuda=True)

    def test_inf_cpu(self):
        self.inf_test(cuda=False)

    def test_inf_gpu(self):
        self.inf_test(cuda=True)

    def test_grad_cpu(self):
        self.grad_test(cuda=False)

    def test_grad_gpu(self):
        self.grad_test(cuda=True)


if __name__ == "__main__":
    unittest.main()
