import numpy as np
import cupy as cp
import time

def read_data(filepath):
    with open(filepath, 'r') as f:
        # Parse header line
        header = f.readline().split()
        rows = int(header[0])
        cols = int(header[1])
        # header[2] is ignored
        n_tumor = int(header[3])
        n_non_tumor = int(header[4])
        # assert n_tumor + n_non_tumor == cols, (
        #     f"tumor ({n_tumor}) + non_tumor ({n_non_tumor}) != cols ({cols})"
        # )

        tumor_matrix = np.empty((rows, n_tumor), dtype=np.uint8)
        non_tumor_matrix = np.empty((rows, n_non_tumor), dtype=np.uint8)

        for i, line in enumerate(f):
            line = line.strip()
            row = (np.frombuffer(line.encode(), dtype=np.uint8) - ord('0')).astype(np.int16)
            tumor_matrix[i] = row[:n_tumor]
            non_tumor_matrix[i] = row[n_tumor:]

    return tumor_matrix, non_tumor_matrix


tumor, normal = read_data('data/BLCA.txt')
print(f"Tumor matrix shape:     {tumor.shape}")
print(f"Non-tumor matrix shape: {normal.shape}")

iteration = 0
total_start = time.time()
while tumor.shape[1] > 0:
    iteration += 1
    print(f"\n=== Iteration {iteration} ===")
    print(f"Tumor matrix shape: {tumor.shape}")

    tumor_t = tumor.T
    tumor_gpu = cp.asarray(tumor, dtype=cp.int16)
    tumor_t_gpu = cp.asarray(tumor_t, dtype=cp.int16)
    mtx_2h_tumor = cp.asnumpy(tumor_gpu @ tumor_t_gpu)

    normal_gpu = cp.asarray(normal, dtype=cp.int16)

    # Extract upper triangle (i < j)
    i_idx, j_idx = np.triu_indices(mtx_2h_tumor.shape[0], k=1)
    upper_vals = mtx_2h_tumor[i_idx, j_idx]

    # Sort by value descending, take top 10000
    sort_order_2h = np.argsort(upper_vals)[::-1]
    sorted_vals_2h = upper_vals[sort_order_2h]
    sorted_indices = np.stack([i_idx[sort_order_2h], j_idx[sort_order_2h]], axis=1)

    print(f"max value of 2 hit is {sorted_vals_2h[0]}")

    top_10k = sorted_indices[:25000]
    boundary_2h = sorted_vals_2h[25000]

    top_i = cp.asarray(top_10k[:, 0])
    top_j = cp.asarray(top_10k[:, 1])

    # For each pair (i,j) and each tumor k: tumor[i,k] & tumor[j,k] -> (10000, 309)
    mtx_10k_tumor = tumor_gpu[top_i] * tumor_gpu[top_j]
    mtx_10k_normal = normal_gpu[top_i] * normal_gpu[top_j]
    print(mtx_10k_normal.shape)

    # (10000, 309) @ (309, 10000) -> (10000, 10000)
    mtx_4h_tumor = mtx_10k_tumor @ mtx_10k_tumor.T
    mtx_4h_normal = mtx_10k_normal @ mtx_10k_normal.T
    mtx_4h = mtx_4h_tumor - (10 * mtx_4h_normal)
    ri, rj = cp.triu_indices(mtx_4h.shape[0], k=1)
    upper_4h = mtx_4h[ri, rj]

    # Filter out entries where a,b,c,d are not 4 distinct original indices
    top_10k_gpu = cp.asarray(top_10k)
    vals = cp.stack([
        top_10k_gpu[ri, 0], top_10k_gpu[ri, 1],
        top_10k_gpu[rj, 0], top_10k_gpu[rj, 1],
    ], axis=1)
    vals_sorted = cp.sort(vals, axis=1)
    valid = cp.all(cp.diff(vals_sorted, axis=1) != 0, axis=1)

    valid_vals = upper_4h[valid]
    valid_positions = cp.where(valid)[0]
    max_valid_flat = cp.argmax(valid_vals)
    orig_idx = valid_positions[max_valid_flat]
    max_val = valid_vals[max_valid_flat]
    max_idx = (int(ri[orig_idx]), int(rj[orig_idx]))
    print(f"Max value (no redundants): {max_val}")
    print(f"Original indices: {top_10k[max_idx[0]]} {top_10k[max_idx[1]]}")

    max_val_cpu = int(max_val)

    if max_val_cpu >= boundary_2h:
        print(f"max_val ({max_val_cpu}) >= boundary_2h ({boundary_2h}), pruning samples...")
        idx_a, idx_b = top_10k[max_idx[0]]
        idx_c, idx_d = top_10k[max_idx[1]]
        mask_tumor = (tumor[idx_a] & tumor[idx_b] & tumor[idx_c] & tumor[idx_d]).astype(bool)
        print(f"Removing {np.sum(mask_tumor)} tumors where indices {idx_a},{idx_b},{idx_c},{idx_d} are all 1")
        tumor = tumor[:, ~mask_tumor]
        
    else:
        print(f"max_val ({max_val_cpu}) < boundary_2h ({boundary_2h}), stopping.")
        break

print(f"\nFinal tumor matrix shape: {tumor.shape}")
print(f"Total loop time: {time.time() - total_start:.2f}s")
