import random
import sys
import time

import torch
import linear_sd_cpp

random.seed(42)
torch.manual_seed(42)


def run_pytorch(sorted_values, ub):
    assert torch.all(sorted_values < ub)
    counts = torch.bincount(sorted_values, minlength=ub)
    ends = torch.cumsum(counts, 0)
    begins = torch.zeros_like(ends)
    begins[1:] = ends[:-1]
    begins[counts == 0] = len(sorted_values)
    ends[counts == 0] = len(sorted_values)
    return begins, ends


def run_cuda(sorted_values, ub):
    return linear_sd_cpp.element_ranges(sorted_values, ub)


def compare(pt, cu, name):
    success = torch.all(pt == cu)
    if not success:
        mismatch = pt != cu
        mismatch_indices = torch.nonzero(mismatch).view(-1)
        sys.stderr.write(f'Mismatches in {name} at the following {torch.sum(mismatch)} indices:\n')
        sys.stderr.write(f'{mismatch_indices}\n')
        sys.stderr.write('Torch outputs:\n')
        sys.stderr.write(f'{pt[mismatch]}\n')
        sys.stderr.write('CUDA outputs:\n')
        sys.stderr.write(f'{cu[mismatch]}\n')
    return success


def run_test(sorted_values, ub):
    start_time1 = time.time()
    begins1, ends1 = run_pytorch(sorted_values, ub)
    end_time1 = time.time()

    start_time2 = time.time()
    begins2, ends2 = run_cuda(sorted_values, ub)
    end_time2 = time.time()

    success = compare(begins1, begins2, 'begins') and compare(ends1, ends2, 'ends')

    return success, end_time1 - start_time1, end_time2 - start_time2


def generate_test(length, ub):
    return torch.sort(torch.randint(0, ub, [length], device='cuda:0', dtype=torch.int64))[0], ub


total_py = 0.0
total_cuda = 0.0
passed = 0
num_tests = 100
for test_num in range(num_tests):
    test = generate_test(random.randrange(1, 2**20), random.randrange(1, 2**16))
    success, py_time, cuda_time = run_test(*test)
    if success:
        passed += 1
        status = "OK"
    else:
        status = "FAILED"
    sys.stderr.write(f'Test {test_num + 1}: {status} (py: {py_time:.8f}s, cuda: {cuda_time:.8f}s)\n')
    sys.stderr.flush()
    if test_num > 0:
        total_py += py_time
        total_cuda += cuda_time

sys.stderr.write(f'{passed}/{num_tests} tests passed (py: {total_py:.8f}, cuda: {total_cuda:.8f})\n')
    
