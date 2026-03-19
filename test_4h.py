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
    """Find best valid 4-hit combo from score matrix.
    mtx_4h: (n_a, n_b) score matrix on GPU
    chunk_a_indices: (n_a, 2) gene indices on GPU (row side)
    chunk_b_indices: (n_b, 2) gene indices on GPU (col side)
    cross: if False, matrix is symmetric -> upper triangle only
    Returns (max_val, g1, g2, g3, g4) or None.
    """
    n_a, n_b = mtx_4h.shape

    # Check 4 distinct genes via broadcasting (2 from row × 2 from col)
    overlap = cp.zeros((n_a, n_b), dtype=bool)
    for k in range(2):
        for l in range(2):
            overlap |= (chunk_a_indices[:, k:k+1] == chunk_b_indices[None, :, l])
    valid = ~overlap

    if not cross:
        valid = cp.triu(valid, k=1)

    if not cp.any(valid):
        return None

    mtx_masked = cp.where(valid, mtx_4h, cp.finfo(mtx_4h.dtype).min)
    flat_idx = int(cp.argmax(mtx_masked))
    p = flat_idx // n_b
    q = flat_idx % n_b
    max_val = int(mtx_4h[p, q])

    genes_out = list(cp.asnumpy(chunk_a_indices[p])) + list(cp.asnumpy(chunk_b_indices[q]))
    return (max_val, *genes_out)


tumor, normal = read_data('data/BLCA.txt')
print(f"Tumor matrix shape:     {tumor.shape}")
print(f"Non-tumor matrix shape: {normal.shape}")

def print_results(results):
    print("\n=== Summary ===")
    for r in results:
        print(f"Iter {r['iter']}: genes ({r['genes'][0]},{r['genes'][1]},{r['genes'][2]},{r['genes'][3]})  "
              f"score={r['score']}  removed={r['removed']}  normals_covered={r['normals_covered']}")

def print_result_genes(results):
    print("\n=== Genes ===")
    for r in results:
        print(f"{r['genes'][0]},{r['genes'][1]},{r['genes'][2]},{r['genes'][3]}")

top_k = 10000
iteration = 0
results = []

tumor_gpu = cp.asarray(tumor, dtype=cp.float16)
normal_gpu = cp.asarray(normal, dtype=cp.float16)

total_start = time.time()

while tumor_gpu.shape[1] > 0:
    iteration += 1
    print(f"\n=== Iteration {iteration} ===")
    print(f"Tumor matrix shape: {tumor_gpu.shape}")

    mtx_2h_tumor = tumor_gpu @ tumor_gpu.T

    # Extract upper triangle (i < j)
    i_idx, j_idx = cp.triu_indices(mtx_2h_tumor.shape[0], k=1) # TODO: cp.triu_indices
    upper_vals = mtx_2h_tumor[i_idx, j_idx]

    # Sort by value descending
    sort_order_2h = cp.argsort(upper_vals)[::-1] # TODO: cp.argsort
    sorted_vals_2h = upper_vals[sort_order_2h]
    sorted_indices = cp.stack([i_idx[sort_order_2h], j_idx[sort_order_2h]], axis=1)

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
        chunk_i = chunk_indices[:, 0]
        chunk_j = chunk_indices[:, 1]

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
        mask_tumor = (tumor_gpu[idx_a] * tumor_gpu[idx_b] * tumor_gpu[idx_c] * tumor_gpu[idx_d]).astype(bool)
        removed = cp.sum(mask_tumor)
        mask_normal = (normal_gpu[idx_a] * normal_gpu[idx_b] * normal_gpu[idx_c] * normal_gpu[idx_d]).astype(bool)
        covered = cp.sum(mask_normal)
        if removed == 0:
            print(f"No tumors removed with genes {idx_a},{idx_b},{idx_c},{idx_d}, stopping.")
            break
        print(f"Removing {removed} tumors ({covered} normals are covered) where indices {idx_a},{idx_b},{idx_c},{idx_d} are all 1")
        results.append({
            'iter': iteration,
            'genes': [int(idx_a), int(idx_b), int(idx_c), int(idx_d)],
            'score': best_val,
            'removed': int(removed),
            'normals_covered': int(covered),
        })
        tumor_gpu = tumor_gpu[:, ~mask_tumor]
    else:
        print("No valid 4-hit combination found, stopping.")
        break

print(f"\nFinal tumor matrix shape: {tumor_gpu.shape}")
print(f"Total loop time: {time.time() - total_start:.2f}s")
print_result_genes(results)
