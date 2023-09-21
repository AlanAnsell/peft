#include <cassert>
#include <optional>

#include <pybind11/stl.h>
#include <torch/torch.h>

namespace py = pybind11;
//#include "scatter.h"


torch::Tensor linear_sd_cuda_backward(
    torch::Tensor input,
    torch::Tensor output_grad,
    torch::Tensor di
);


class LinearWithSparseDelta : public torch::autograd::Function<LinearWithSparseDelta> {
  public:

    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor dv,
        torch::Tensor di,
        std::optional<torch::Tensor> bias
    ) {
        if (bias)
            ctx->save_for_backward({input, weight, dv, di, *bias});
        else
            ctx->save_for_backward({input, weight, dv, di});
        torch::Tensor W = weight.clone();
        //scatter_sum_coo(dv, di, W.flatten());
        W.view(-1).scatter_add_(0, di, dv.to(W.dtype()));
        return torch::nn::functional::linear(
            input,
            W,
            bias ? *bias : torch::Tensor()
        );
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto dv = saved[2];
        auto di = saved[3];
        torch::Tensor bias;
        if (saved.size() > 4)
            bias = saved[4];
        auto output_grad = grad_outputs[0];

        torch::Tensor W = weight.clone();
        W.view(-1).scatter_add_(0, di, dv.to(W.dtype()));
        //scatter_sum_coo(dv, di, W.flatten());

        torch::Tensor input_grad, weight_grad, dv_grad, bias_grad;
        torch::Tensor output_grad_2d = output_grad.reshape({-1, output_grad.size(-1)});

        if (ctx->needs_input_grad(0))
            input_grad = output_grad_2d.mm(W).view_as(input);

        //if (ctx->needs_input_grad(1) ||
        //        (ctx->needs_input_grad(2) && ! dv.device().is_cuda())) {
            torch::Tensor input_2d = input.reshape({-1, input.size(-1)});
            weight_grad = output_grad_2d.t().mm(input_2d);
            //dv_grad = gather_coo(weight_grad.flatten(), di);
            if (ctx->needs_input_grad(2))
                dv_grad = weight_grad.view(-1).gather(0, di).to(dv.dtype());
        //} else if (ctx->needs_input_grad(2)) {
        //    torch::Tensor input_2d = input.reshape({-1, input.size(-1)});
        //    dv_grad = linear_sd_cuda_backward(
        //        input_2d.t().contiguous(),
        //        output_grad_2d.t().contiguous(),
        //        di
        //    );
        //}

        if (bias.defined() && ctx->needs_input_grad(4))
            bias_grad = output_grad_2d.sum(0);

        return {input_grad, weight_grad, dv_grad, torch::Tensor(), bias_grad};
    }

};

torch::Tensor apply_linear_sd(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor dv,
    torch::Tensor di,
    std::optional<torch::Tensor> bias = std::nullopt
) {
    return LinearWithSparseDelta::apply(
        input,
        weight,
        dv,
        di,
        bias
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "apply",
        &apply_linear_sd,
        py::arg("input"),
        py::arg("weight"),
        py::arg("dv"),
        py::arg("di"),
        py::arg_v("bias", std::nullopt)
    );
}
