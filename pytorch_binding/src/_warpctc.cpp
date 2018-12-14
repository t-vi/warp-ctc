// PyTorch bindings (c) 2018 by Thomas Viehmann <tv@lernapparat.de>
// All rights reserved. Licensed under the
// Apache License,  Version 2.0, January 2004
// see LICENSE in root directory



#include <torch/extension.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAStream.h>
#include <tuple>
#include <iostream>
#include "ctc.h"

// pytorch 0.4 compatibility
#ifndef AT_CHECK
#define AT_CHECK AT_ASSERT
#endif

std::tuple<at::Tensor, at::Tensor> ctc(at::Tensor activations,
				       at::Tensor input_lengths,
				       at::Tensor labels,
				       at::Tensor label_lengths,
				       int blank_label = 0,
				       bool want_gradients = true)
{
  try {
    auto is_cuda = activations.type().is_cuda();

    auto activations_arg = at::TensorArg(activations, "activations", 0);
    checkScalarType("activations", activations_arg, at::kFloat);
    checkContiguous("activations", activations_arg);
    checkDim("activations", activations_arg, 3);

    auto input_lengths_arg = at::TensorArg(input_lengths, "input_lengths", 1);
    checkScalarType("input_lengths", input_lengths_arg, at::kInt);
    checkContiguous("input_lengths", input_lengths_arg);
    checkDim("input_lengths", input_lengths_arg, 1);
    AT_CHECK(! input_lengths.type().is_cuda(), "input_lengths must be on CPU");

    auto labels_arg = at::TensorArg(labels, "labels", 2);
    checkScalarType("labels", labels_arg, at::kInt);
    checkContiguous("labels", labels_arg);
    checkDim("labels", labels_arg, 1);
    AT_CHECK(! labels.type().is_cuda(), "labels must be on CPU");

    auto label_lengths_arg = at::TensorArg(label_lengths, "label_lengths", 3);
    checkScalarType("label_lengths", label_lengths_arg, at::kInt);
    checkContiguous("label_lengths", label_lengths_arg);
    checkDim("label_lengths", label_lengths_arg, 1);
    AT_CHECK(! label_lengths.type().is_cuda(), "label_lengths must be on CPU");

    const auto batch_size = activations.size(1);
    const auto alphabet_size = activations.size(2);
    checkSize("input_lengths", input_lengths_arg, 0, batch_size);
    checkSize("label_lengths", label_lengths_arg, 0, batch_size);
    const auto total_label_length = label_lengths.toType(at::kLong).sum().item<int64_t>();
    checkSize("labels", labels_arg, 0, total_label_length);

    ctcOptions options{};
    options.blank_label = blank_label;
    if (! is_cuda) {
        options.loc = CTC_CPU;
        options.num_threads = 0; // will use default number of threads

        #if defined(CTC_DISABLE_OMP) || defined(APPLE)
        // have to use at least one
        options.num_threads = std::max(options.num_threads, (unsigned int) 1);
        #endif
    }
    else {
        options.loc = CTC_GPU;
        options.stream = at::cuda::getCurrentCUDAStream();
    }
   
   
/*
ctcStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size, int minibatch,
                               ctcOptions info,
                               size_t* size_bytes);

*/

    size_t workspace_size;
    ctcStatus_t status;
    status = get_workspace_size(label_lengths.data<int>(), input_lengths.data<int>(),
				(int) alphabet_size, batch_size,
				options, &workspace_size);

    if (status != CTC_STATUS_SUCCESS) {
       AT_ERROR("warp_ctc error: ", ctcGetStatusString(status));
    }
   
    at::Tensor workspace = torch::empty({(int64_t) workspace_size}, activations.options().dtype(at::kByte));

    at::Tensor costs = torch::empty({batch_size}, labels.options().dtype(at::kFloat)); // always CPU
    at::Tensor gradients;
    if (want_gradients) // we need to initialize to 0 to avoid NaNs in the "unused parts"
      gradients = torch::zeros_like(activations);
    else
      gradients = torch::zeros({0}, activations.options());

    status = compute_ctc_loss(activations.data<float>(), (want_gradients ? gradients.data<float>() : NULL),
			      labels.data<int>(), label_lengths.data<int>(),
			      input_lengths.data<int>(), alphabet_size,
			      batch_size, costs.data<float>(),
			      workspace.data_ptr(), options);

    if (status != CTC_STATUS_SUCCESS) {
       AT_ERROR("warp_ctc error: ", ctcGetStatusString(status));
    }
    return std::make_tuple(costs, gradients);

   /*
ctcStatus_t compute_ctc_loss(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             ctcOptions options);
*/
  } catch (const at::Error &e) {
    throw (torch::ValueError(e.what_without_backtrace()));
  }
}


PYBIND11_MODULE(_warpctc, m) {
  m.attr("__name__") = "warpctc._warpctc";
  m.def("ctc", &ctc, "CTC");
}
