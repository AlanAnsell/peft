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

torch::Tensor linear_sd_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor dv,
    torch::Tensor di,
    std::optional<torch::Tensor> bias
) {
    torch::Tensor W = weight.clone();
    //scatter_sum_coo(dv, di, W.flatten());
    W.reshape(-1).scatter_add_(0, di, dv.to(W.dtype()));
    return torch::nn::functional::linear(
        input.to(W.dtype()),
        W,
        bias ? *bias : torch::Tensor()
    );
}

torch::autograd::tensor_list linear_sd_backward(
    torch::Tensor output_grad,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor dv,
    torch::Tensor di,
    bool input_needs_grad,
    bool weight_needs_grad,
    bool dv_needs_grad,
    bool bias_needs_grad,
    std::optional<torch::Tensor> bias = std::nullopt
) {
    torch::Tensor W = weight.clone();
    W.reshape(-1).scatter_add_(0, di, dv.to(W.dtype()));
    //scatter_sum_coo(dv, di, W.flatten());

    torch::Tensor input_grad, weight_grad, dv_grad, bias_grad;
    torch::Tensor output_grad_2d = output_grad.reshape({-1, output_grad.size(-1)});

    if (input_needs_grad)
        input_grad = output_grad_2d.mm(W.to(output_grad_2d.dtype())).view_as(input);

    if (weight_needs_grad) {
        torch::Tensor input_2d = input.reshape({-1, input.size(-1)});
        weight_grad = output_grad_2d.t().mm(input_2d.to(output_grad_2d.dtype()));
        //dv_grad = gather_coo(weight_grad.flatten(), di);
        if (dv_needs_grad)
            dv_grad = weight_grad.view(-1).gather(0, di).to(dv.dtype());
    } else if (dv_needs_grad) {
        torch::Tensor input_2d = input.reshape({-1, input.size(-1)});
        dv_grad = linear_sd_cuda_backward(
            //input_2d.t().contiguous(),
            //output_grad_2d.t().contiguous(),
            input_2d,
            output_grad_2d,
            di
        );
        //torch::Tensor rows = di.floor_divide(weight.size(1));
        //torch::Tensor columns = di - weight.size(1) * rows;

        //torch::Tensor input_2d = input.reshape({-1, input.size(-1)}).t().to(dv.dtype()).contiguous();
        //dv_grad = torch::linalg_vecdot(
        //    input_2d.index({columns, torch::indexing::Slice()}),
        //    output_grad_2d.t().to(dv.dtype()).contiguous().index({rows, torch::indexing::Slice()}),
        //    1
        //);
    }

    if (bias && bias_needs_grad)
        bias_grad = output_grad_2d.sum(0);

    return {input_grad, weight_grad, dv_grad, torch::Tensor(), bias_grad};
}

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
        return linear_sd_forward(input, weight, dv, di, bias);
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
        auto output_grad = grad_outputs[0];

        return linear_sd_backward(
            output_grad,
            input,
            weight,
            dv,
            di,
            ctx->needs_input_grad(0),
            ctx->needs_input_grad(1),
            ctx->needs_input_grad(2),
            saved.size() > 4 && ctx->needs_input_grad(4),
            saved.size() > 4 ?
                std::optional<torch::Tensor>(saved[4]) :
                std::optional<torch::Tensor>(std::nullopt)
        );
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
        "forward",
        &linear_sd_forward
    );
    m.def(
        "backward",
        &linear_sd_backward
    );
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
