import numpy as np
import cupy as cp

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
            row = np.frombuffer(line.encode(), dtype=np.uint8) - ord('0')
            tumor_matrix[i] = row[:n_tumor]
            non_tumor_matrix[i] = row[n_tumor:]

    return tumor_matrix, non_tumor_matrix


tumor, non_tumor = read_data('data/BLCA.txt')
tumor_t = tumor.T
tumor_gpu = cp.asarray(tumor)
tumor_t_gpu = cp.asarray(tumor_t)
mtx_2h = cp.asnumpy(tumor_gpu @ tumor_t_gpu)
print(f"Tumor matrix shape:     {tumor.shape}")
print(f"Non-tumor matrix shape: {non_tumor.shape}")
print(f"Tumor transposed shape: {tumor_t.shape}")

# Extract upper triangle (i < j)
i_idx, j_idx = np.triu_indices(mtx_2h.shape[0], k=1)
upper_vals = mtx_2h[i_idx, j_idx]

# Sort by value descending, take top 10000
sort_order_2h = np.argsort(upper_vals)[::-1]
sorted_vals_2h = upper_vals[sort_order_2h]
sorted_indices = np.stack([i_idx[sort_order_2h], j_idx[sort_order_2h]], axis=1)

print(f"max value of 2 hit is {sorted_vals_2h[0]}")

top_10k = sorted_indices[:10000]
print(f"test top_10k: {top_10k[0]}")
top_i = cp.asarray(top_10k[:, 0])
top_j = cp.asarray(top_10k[:, 1])

# For each pair (i,j) and each tumor k: tumor[i,k] & tumor[j,k] -> (10000, 309)
mtx_10k = tumor_gpu[top_i] * tumor_gpu[top_j]  # (10000, 309)
print(f"mtx_10k shape: {mtx_10k.shape}")

# (10000, 309) @ (309, 10000) -> (10000, 10000)
mtx_4h = mtx_10k @ mtx_10k.T
ri, rj = cp.triu_indices(mtx_4h.shape[0], k=1)
upper_4h = mtx_4h[ri, rj]

# Each entry (i,j) combines top_10k[i] = (a,b) and top_10k[j] = (c,d)
# Filter out entries where a,b,c,d are not 4 distinct original indices
top_10k_gpu = cp.asarray(top_10k)  # (10000, 2)
vals = cp.stack([
    top_10k_gpu[ri, 0], top_10k_gpu[ri, 1],
    top_10k_gpu[rj, 0], top_10k_gpu[rj, 1],
], axis=1)  # (N, 4)
vals_sorted = cp.sort(vals, axis=1)
valid = cp.all(cp.diff(vals_sorted, axis=1) != 0, axis=1)

valid_vals = upper_4h[valid]
valid_positions = cp.where(valid)[0]
max_valid_flat = cp.argmax(valid_vals)
orig_idx = valid_positions[max_valid_flat]
max_val = valid_vals[max_valid_flat]
max_idx = (int(ri[orig_idx]), int(rj[orig_idx]))
print(f"Max value (no redundants): {max_val}")
print(f"Max index in mtx_4h: ({max_idx[0]}, {max_idx[1]})")
print(f"Original indices: {top_10k[max_idx[0]]} {top_10k[max_idx[1]]}")

max_val_cpu = int(max_val)
count_larger = np.sum(sorted_vals_2h > max_val_cpu)
count_nonzero_less = np.sum((sorted_vals_2h < max_val_cpu) & (sorted_vals_2h > 0))
count_zero = np.sum(sorted_vals_2h == 0)
print(f"Larger than max_val: {count_larger}")
print(f"Non-zero less than max_val: {count_nonzero_less}")
print(f"Zeros: {count_zero}")
