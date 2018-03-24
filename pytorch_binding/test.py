#!/usr/bin/python3

import torch
import warpctc

# tests from warpctc/test/test_cpu.cpp

def small_test(cuda = False):
    alphabet_size = 5
    T = 2
    print ("small_test with cuda={}".format(cuda))
    activations = torch.FloatTensor(
        [[0.1, 0.6, 0.1, 0.1, 0.1],
         [0.1, 0.1, 0.6, 0.1, 0.1]]).unsqueeze(1)
    if cuda:
        activations = activations.cuda()
    # Score calculation is specific to the given activations above
    probs = torch.nn.functional.softmax(activations, -1)
    expected_score = probs[0, 0, 1] * probs[1, 0, 2]

    labels = torch.IntTensor([1,2])
    label_lengths = torch.IntTensor([2])
    lengths = torch.IntTensor([T])
    
    score = warpctc.CTCFunction.apply(activations, lengths, labels, label_lengths)

    score = torch.exp(-score)
    eps = 1e-6

    assert (expected_score-eps < score < expected_score + eps), "Test failed with cuda={}".format(cuda)
    print ("OK")

def options_test(cuda):
    print ("options_test with cuda={}".format(cuda))
    alphabet_size = 6
    T = 5
    minibatch = 2
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
         [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]], requires_grad=True)

    expected_grads = torch.FloatTensor([ # from test_cpu from tensorflow
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
             -torch.log(activations[0,0,0]*activations[1,0,1]*activations[2,0,2]
                        *activations[3,0,1]*activations[4,0,0]),
             5.42262 # from tensorflow
             ])

    if cuda:
        activations = activations.cuda()
        expected_scores = expected_scores.cuda()
        expected_grads = expected_grads.cuda()

    activations = torch.log(activations)
    activations.retain_grad()

    labels = torch.IntTensor([0, 1, 2, 1, 0,
                              0, 1, 1, 0])

    label_lengths = torch.IntTensor([5, 4])
    lengths = torch.IntTensor([5, 5])

    scores = warpctc.CTCFunction.apply(activations, lengths, labels, label_lengths, 5) # last argument is blank_label

    scores.sum().backward()

    eps = 1e-4
    assert (scores-expected_scores).abs().max()<eps, "Scores failed with cuda={}".format(cuda)
    assert (activations.grad-expected_grads).abs().max()<eps, "Scores failed with cuda={}".format(cuda)
    print ("OK")


def inf_test(cuda = False):
    print ("inf_test with cuda={}".format(cuda))
    alphabet_size = 15
    T = 50
    L = 10
    minibatch = 1

    labels = torch.IntTensor(L).random_(1, alphabet_size)
    if L >= 3: # guarantee repeats for testing
        labels[L // 2]     = labels[L // 2 + 1]
        labels[L // 2 - 1] = labels[L // 2]
    labels[0] = 2;
    
    label_lengths = torch.IntTensor([L])

    acts = torch.rand(T, minibatch, alphabet_size)
    acts[:, 0, 2] = -1e30

    if cuda:
        acts = acts.cuda()

    acts = torch.tensor(acts, requires_grad=True, dtype=acts.dtype)

    sizes = torch.IntTensor([T])

    cost = warpctc.CTCFunction.apply(acts, sizes, labels, label_lengths)

    assert cost == float("inf"), "cost should be inf"

    cost.sum().backward()
    
    assert (acts.grad == acts.grad).all(), "gradient of inf cost should not be NaN" # g==g is true unless it is NaN
    print("OK")

def grad_test(cuda = False):
    print ("grad_test with cuda={}".format(cuda))
    problem_sizes = [[20, 50, 15, 1, 10**(-2.5)],
                     [5, 10, 5, 65, 1e-2]]   # n.b. warpctc's tests use squared relative difference

    for alphabet_size, T, L, minibatch, tol in problem_sizes:
        acts = torch.rand(T, minibatch, alphabet_size, requires_grad=True, dtype=(torch.cuda.float32 if cuda else torch.float32))
        sizes =  torch.IntTensor(minibatch).fill_(T)
        label_lengths = torch.IntTensor(minibatch).fill_(L)
        labels = torch.IntTensor(L, minibatch).random_(1, alphabet_size)
        if L >= 3: # guarantee repeats for testing
            labels[L // 2]     = labels[L // 2 + 1]
            labels[L // 2 - 1] = labels[L // 2]
        labels = labels.view(-1)
        torch.autograd.gradcheck(warpctc.CTCFunction.apply, (acts, sizes, labels, label_lengths), rtol=tol, atol=tol, eps=1e-1)
    
    print ("OK")
    
if __name__ == "__main__":
    small_test(cuda=False)
    small_test(cuda=True)
    options_test(cuda=False)
    options_test(cuda=True)
    inf_test(cuda=False)
    inf_test(cuda=True)
    grad_test(cuda=False)
    grad_test(cuda=True)

