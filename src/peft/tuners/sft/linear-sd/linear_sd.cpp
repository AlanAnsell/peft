#include <torch/torch.h>

namespace py = pybind11;
//#include "scatter.h"

//torch::Tensor linear_sd_forward(
//    torch::Tensor input,
//    torch::Tensor weight,
//    torch::Tensor dv,
//    torch::Tensor di,
//    torch::Tensor bias = torch::Tensor()
//) {
//    torch::Tensor W = weight.clone();
//    //scatter_sum_coo(dv, di, W.flatten());
//    W.flatten().scatter_add_(0, di, dv);
//    return torch::nn::functional::linear(input, W, bias);
//}
//
//static torch::autograd::tensor_list backward(
//    torch::Tensor input,
//    torch::Tensor weight,
//    torch::Tensor dv,
//    torch::Tensor di,
//    torch::Tensor bias = torch::Tensor()
//) {
//    auto saved = ctx->get_saved_variables();
//    auto input = saved[0];
//    auto weight = saved[1];
//    auto dv = saved[2];
//    auto di = saved[3];
//    auto bias = saved[4];
//    auto output_grad = grad_outputs[0];
//
//    torch::Tensor W = weight.clone();
//    W.flatten().scatter_add_(0, di, dv);
//    //scatter_sum_coo(dv, di, W.flatten());
//
//    torch::Tensor input_grad, weight_grad, dv_grad, bias_grad;
//    if (ctx->needs_input_grad(0))
//        input_grad = output_grad.mm(W);
//
//    weight_grad = input.t().mm(output_grad.t());
//    //dv_grad = gather_coo(weight_grad.flatten(), di);
//    dv_grad = weight_grad.flatten().gather(0, di);
//
//    if (ctx->needs_input_grad(4))
//        bias_grad = output_grad.sum(0);
//
//    return {input_grad, weight_grad, dv_grad, torch::Tensor(), bias_grad};
//}

class LinearWithSparseDelta : public torch::autograd::Function<LinearWithSparseDelta> {
  public:

    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor dv,
        torch::Tensor di,
        torch::Tensor bias = torch::Tensor()
    ) {
        ctx->save_for_backward({input, weight, dv, di, bias});
        torch::Tensor W = weight.clone();
        //scatter_sum_coo(dv, di, W.flatten());
        W.view(-1).scatter_add_(0, di, dv);
        return torch::nn::functional::linear(input, W, bias);
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
        auto bias = saved[4];
        auto output_grad = grad_outputs[0];

        torch::Tensor W = weight.clone();
        W.view(-1).scatter_add_(0, di, dv);
        //scatter_sum_coo(dv, di, W.flatten());

        torch::Tensor input_grad, weight_grad, dv_grad, bias_grad;
        if (ctx->needs_input_grad(0))
            input_grad = output_grad.mm(W);

        weight_grad = output_grad.t().mm(input);
        //dv_grad = gather_coo(weight_grad.flatten(), di);
        dv_grad = weight_grad.view(-1).gather(0, di);

        //if (ctx->needs_input_grad(4))
        //    bias_grad = output_grad.sum(0);

        return {input_grad, weight_grad, dv_grad, torch::Tensor(), bias_grad};
    }

};

torch::Tensor apply_linear_sd(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor dv,
    torch::Tensor di
    //torch::Tensor bias = torch::Tensor()
) {
    return LinearWithSparseDelta::apply(input, weight, dv, di);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply", &apply_linear_sd, "LinearSD");
}
