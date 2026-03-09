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

        tumor_matrix = np.empty((rows, n_tumor), dtype=np.uint8)
        non_tumor_matrix = np.empty((rows, n_non_tumor), dtype=np.uint8)

        for i, line in enumerate(f):
            line = line.strip()
            row = (np.frombuffer(line.encode(), dtype=np.uint8) - ord('0')).astype(np.int16)
            tumor_matrix[i] = row[:n_tumor]
            non_tumor_matrix[i] = row[n_tumor:]

    return tumor_matrix, non_tumor_matrix


def find_best_valid_4h(mtx_4h, chunk_a_indices, chunk_b_indices, cross):
    """Find the best valid 4-hit combo from a 4-hit score matrix.

    For cross=False (same chunk): use upper triangle only.
    For cross=True (different chunks): use all entries.

    Returns (max_val, gene_a, gene_b, gene_c, gene_d) or None if no valid entry.
    """
    if cross:
        # All entries: row from chunk_a, col from chunk_b
        ri, rj = cp.meshgrid(cp.arange(mtx_4h.shape[0]), cp.arange(mtx_4h.shape[1]), indexing='ij')
        ri = ri.ravel()
        rj = rj.ravel()
        flat_vals = mtx_4h.ravel()
    else:
        ri, rj = cp.triu_indices(mtx_4h.shape[0], k=1)
        flat_vals = mtx_4h[ri, rj]

    chunk_a_gpu = cp.asarray(chunk_a_indices)
    chunk_b_gpu = cp.asarray(chunk_b_indices)

    vals = cp.stack([
        chunk_a_gpu[ri, 0], chunk_a_gpu[ri, 1],
        chunk_b_gpu[rj, 0], chunk_b_gpu[rj, 1],
    ], axis=1)
    vals_sorted = cp.sort(vals, axis=1)
    valid = cp.all(cp.diff(vals_sorted, axis=1) != 0, axis=1)

    if not cp.any(valid):
        return None

    valid_vals = flat_vals[valid]
    valid_positions = cp.where(valid)[0]
    max_valid_flat = cp.argmax(valid_vals)
    orig_idx = valid_positions[max_valid_flat]
    max_val = int(valid_vals[max_valid_flat])

    i_in_chunk = int(ri[orig_idx])
    j_in_chunk = int(rj[orig_idx])
    gene_a, gene_b = chunk_a_indices[i_in_chunk]
    gene_c, gene_d = chunk_b_indices[j_in_chunk]

    return (max_val, gene_a, gene_b, gene_c, gene_d)


tumor, normal = read_data('data/HNSC.txt')
print(f"Tumor matrix shape:     {tumor.shape}")
print(f"Non-tumor matrix shape: {normal.shape}")

top_k = 10000
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

    # Sort by value descending
    sort_order_2h = np.argsort(upper_vals)[::-1]
    sorted_vals_2h = upper_vals[sort_order_2h]
    sorted_indices = np.stack([i_idx[sort_order_2h], j_idx[sort_order_2h]], axis=1)

    print(f"max value of 2 hit is {sorted_vals_2h[0]}")

    total_pairs = len(sorted_vals_2h)
    best_val = None
    best_genes = None

    # Store chunk data for cross-chunk computation
    chunk_tumor_list = []
    chunk_normal_list = []
    chunk_indices_list = []

    n_chunk = 0
    while True:
        chunk_start = n_chunk * top_k
        chunk_end = min((n_chunk + 1) * top_k, total_pairs)

        if chunk_start >= total_pairs:
            print("2-hit table exhausted.")
            break

        boundary_2h = sorted_vals_2h[chunk_end] if chunk_end < total_pairs else 0

        # Get current chunk's pairs
        chunk_indices = sorted_indices[chunk_start:chunk_end]
        chunk_i = cp.asarray(chunk_indices[:, 0])
        chunk_j = cp.asarray(chunk_indices[:, 1])

        # Build 2-hit masks for current chunk
        chunk_tumor = tumor_gpu[chunk_i] * tumor_gpu[chunk_j]
        chunk_normal = normal_gpu[chunk_i] * normal_gpu[chunk_j]

        chunk_tumor_list.append(chunk_tumor)
        chunk_normal_list.append(chunk_normal)
        chunk_indices_list.append(chunk_indices)

        # Compute new chunk × new chunk (upper triangle)
        mtx_4h_tumor_nn = chunk_tumor @ chunk_tumor.T
        mtx_4h_normal_nn = chunk_normal @ chunk_normal.T
        mtx_4h_nn = mtx_4h_tumor_nn - (10 * mtx_4h_normal_nn)

        result = find_best_valid_4h(mtx_4h_nn, chunk_indices, chunk_indices, cross=False)
        if result and (best_val is None or result[0] > best_val):
            best_val, *best_genes = result

        # Compute cross terms: previous chunks × new chunk
        for prev_idx in range(n_chunk):
            mtx_4h_tumor_cross = chunk_tumor_list[prev_idx] @ chunk_tumor.T
            mtx_4h_normal_cross = chunk_normal_list[prev_idx] @ chunk_normal.T
            mtx_4h_cross = mtx_4h_tumor_cross - (10 * mtx_4h_normal_cross)

            result = find_best_valid_4h(mtx_4h_cross, chunk_indices_list[prev_idx], chunk_indices, cross=True)
            if result and (best_val is None or result[0] > best_val):
                best_val, *best_genes = result

        print(f"  Chunk {n_chunk}: pairs [{chunk_start}:{chunk_end}], boundary_2h={boundary_2h}, best_val={best_val}")

        if best_val is not None and best_val >= boundary_2h:
            print(f"  best_val ({best_val}) >= boundary_2h ({boundary_2h}), search complete.")
            break

        n_chunk += 1

    if best_genes is not None:
        idx_a, idx_b, idx_c, idx_d = best_genes
        mask_tumor = (tumor[idx_a] & tumor[idx_b] & tumor[idx_c] & tumor[idx_d]).astype(bool)
        removed = np.sum(mask_tumor)
        if removed == 0:
            print(f"No tumors removed with genes {idx_a},{idx_b},{idx_c},{idx_d}, stopping.")
            break
        print(f"Removing {removed} tumors where indices {idx_a},{idx_b},{idx_c},{idx_d} are all 1")
        tumor = tumor[:, ~mask_tumor]
    else:
        print("No valid 4-hit combination found, stopping.")
        break

print(f"\nFinal tumor matrix shape: {tumor.shape}")
print(f"Total loop time: {time.time() - total_start:.2f}s")
