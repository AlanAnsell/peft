import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import linear_sd_cpp

from peft.tuners.sft.layer import (
    flatten_indices,
    expand_indices,
    random_subset,
    SparseDelta,
)

torch.manual_seed(42)
random.seed(42)

def assertApproxEqual(tensor1, tensor2, name, eps=1e-6):
    if tensor1.size() != tensor2.size():
        sys.stderr.write(f'Got mismatched tensor sizes {tensor1.size()} and {tensor2.size()} for {name}\n')
        return False
    mismatch = torch.abs(tensor1 - tensor2) / (torch.minimum(torch.abs(tensor1), torch.abs(tensor2)) + 1e-4) > eps
    num_mismatches = torch.sum(mismatch)
    if num_mismatches > 0:
        sys.stderr.write(f'{num_mismatches} / {tensor1.numel()} mismatches in {name}:\n')
        mismatch_indices = torch.nonzero(mismatch).view(-1)
        sys.stderr.write(f'Mismatched indices: {mismatch_indices}\n')
        sys.stderr.write(f'{tensor1[mismatch]}\n')
        sys.stderr.write(f'{tensor2[mismatch]}\n')
        return False
    return True


def run_pytorch(x, weight, dv, di, bias=None):
    merged_weight = weight.view(-1) + torch_scatter.segment_coo(
        dv.to(weight.dtype),
        di,
        dim_size=weight.numel(),
        reduce="sum",
    )
    merged_weight = merged_weight.view_as(weight)
    return F.linear(x, merged_weight, bias=bias)


def run_cuda(x, weight, dv, di, bias=None):
    return linear_sd_cpp.apply(x, weight, dv, di, bias=bias)


def prepare_inputs(x, weight, dv, di, bias=None, weight_requires_grad=False, dtype=None):
    if dtype is not None:
        x = x.to(dtype=dtype)
        weight = weight.to(dtype=dtype)
        if bias is not None:
            bias = bias.to(dtype=dtype)
    x = x.detach().cuda()
    x.requires_grad = False
    weight = weight.detach().cuda()
    weight.requires_grad = weight_requires_grad
    dv = dv.detach().cuda()
    dv.requires_grad = True
    di = di.clone().cuda()
    if bias is not None:
        bias = bias.detach().cuda()
        bias.requires_grad = True
    return x, weight, dv, di, bias

def run_test(x, weight, dv, di, bias=None, weight_requires_grad=False, dtype=torch.bfloat16):
    inputs1 = prepare_inputs(x, weight, dv, di, bias, weight_requires_grad, dtype=dtype)
    inputs2 = prepare_inputs(x, weight, dv, di, bias, weight_requires_grad, dtype=dtype)

    output2 = run_cuda(*inputs2)
    loss2 = torch.sum(torch.sigmoid(output2))
    scalar_loss = loss2.item()
    start_time2 = time.time()
    loss2.backward()
    end_time2 = time.time()
    output1 = run_pytorch(*inputs1)
    loss1 = torch.sum(torch.sigmoid(output1))
    scalar_loss = loss1.item()
    start_time1 = time.time()
    loss1.backward()
    end_time1 = time.time()

    success = assertApproxEqual(output1, output2, "output")
    if inputs1[0].requires_grad:
        success = assertApproxEqual(inputs1[0].grad, inputs2[0].grad, "input gradient") and success
    success = assertApproxEqual(inputs1[2].grad, inputs2[2].grad, "dv gradient", eps=1e-2) and success
    if weight_requires_grad:
        success = assertApproxEqual(inputs1[1].grad, inputs2[1].grad, "weight gradient") and success
    if bias is not None:
        success = assertApproxEqual(inputs1[4].grad, inputs2[4].grad, "bias gradient") and success

    if dtype != torch.float32:
        precise_inputs = prepare_inputs(x, weight, dv, di, bias, weight_requires_grad, dtype=torch.float32)
        precise_output = run_pytorch(*precise_inputs)
        precise_loss = torch.sum(torch.sigmoid(precise_output))
        precise_loss.backward()
        py_deviation = torch.sum(torch.abs(precise_inputs[2].grad - inputs1[2].grad)) / precise_inputs[2].numel()
        cuda_deviation = torch.sum(torch.abs(precise_inputs[2].grad - inputs2[2].grad)) / precise_inputs[2].numel()
        sys.stderr.write(f'py_deviation = {py_deviation:.5f}, cuda_deviation = {cuda_deviation:.5f}\n')

    return success, end_time1 - start_time1, end_time2 - start_time2


def generate_test(batch_dims, input_dim, output_dim, density, device=None):
    x = torch.randn(*batch_dims, input_dim, device=device)
    weight = torch.randn(output_dim, input_dim, device=device)
    n = int(density * weight.numel())
    dv = torch.randn(n, device=device) / 10
    di = random_subset(weight.size(), n, device=device)
    di, _ = torch.sort(di)
    #if random.random() < 0.5:
    #    bias = torch.randn(output_dim, dtype=dtype)
    #else:
    bias = None
    return x, weight, dv, di, bias


total_py = 0.0
total_cuda = 0.0
passed = 0
num_tests = 100
for test_num in range(num_tests):
    test = generate_test((8, 256), 4096, 11008, 0.02, device='cuda:0')
    success, py_time, cuda_time = run_test(*test, dtype=torch.float32)
    if success:
        passed += 1
        status = "OK"
    else:
        status = "FAILED"
    sys.stderr.write(f'Test {test_num + 1}: {status} (py: {py_time:.5f}s, cuda: {cuda_time:.5f}s)\n')
    sys.stderr.flush()
    if test_num > 0:
        total_py += py_time
        total_cuda += cuda_time

sys.stderr.write(f'{passed}/{num_tests} tests passed (py: {total_py:.5f}, cuda: {total_cuda:.5f})\n')
    
