import numpy as np
import cupy as cp
import time

def read_data(filepath):
    with open(filepath, 'r') as f:
        header = f.readline().split()
        rows = int(header[0])
        cols = int(header[1])
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


def find_best_valid_6h(mtx_6h, genes_4h, genes_2h):
    """Find best valid 6-hit combo from score matrix.
    mtx_6h: (n_4h, n_2h) score matrix on GPU, NOT symmetric
    genes_4h: (n_4h, 4) gene indices on GPU
    genes_2h: (n_2h, 2) gene indices on GPU
    Returns (max_val, g1, g2, g3, g4, g5, g6) or None.
    """
    n_4h, n_2h = mtx_6h.shape

    # Check 6 distinct genes via broadcasting (memory efficient: one (n_4h, n_2h) bool at a time)
    overlap = cp.zeros((n_4h, n_2h), dtype=bool)
    for k in range(4):
        for l in range(2):
            overlap |= (genes_4h[:, k:k+1] == genes_2h[None, :, l])
    valid = ~overlap

    if not cp.any(valid):
        return None

    mtx_masked = cp.where(valid, mtx_6h, cp.iinfo(mtx_6h.dtype).min)
    flat_idx = int(cp.argmax(mtx_masked))
    p = flat_idx // n_2h
    q = flat_idx % n_2h
    max_val = int(mtx_6h[p, q])

    genes_out = list(cp.asnumpy(genes_4h[p])) + list(cp.asnumpy(genes_2h[q]))
    return (max_val, *genes_out)


tumor, normal = read_data('data/BLCA.txt')
print(f"Tumor matrix shape:     {tumor.shape}")
print(f"Non-tumor matrix shape: {normal.shape}")

def print_results(results):
    print("\n=== Summary ===")
    for r in results:
        genes = ','.join(str(g) for g in r['genes'])
        print(f"Iter {r['iter']}: genes ({genes})  "
              f"score={r['score']}  removed={r['removed']}  normals_covered={r['normals_covered']}")

def print_result_genes(results):
    print("\n=== Genes ===")
    for r in results:
        print(','.join(str(g) for g in r['genes']))

top_k_2h = 25000
top_k_4h = 50000
iteration = 0
results = []
total_start = time.time()

tumor_gpu = cp.asarray(tumor, dtype=cp.int16)
normal_gpu = cp.asarray(normal, dtype=cp.int16)

while tumor_gpu.shape[1] > 0:
    iteration += 1
    print(f"\n=== Iteration {iteration} ===")
    print(f"Tumor matrix shape: {tumor_gpu.shape}")

    # === Step 1: 2-hit ===
    mtx_2h_tp = tumor_gpu @ tumor_gpu.T
    i_idx, j_idx = cp.triu_indices(mtx_2h_tp.shape[0], k=1)
    upper_2h = mtx_2h_tp[i_idx, j_idx]
    del mtx_2h_tp
    sort_2h = cp.argsort(upper_2h)[::-1]
    sorted_2h_vals = upper_2h[sort_2h]
    sorted_2h_indices = cp.stack([i_idx[sort_2h], j_idx[sort_2h]], axis=1)

    total_2h = len(sorted_2h_vals)
    actual_2h = min(top_k_2h, total_2h)
    boundary_2h = int(sorted_2h_vals[actual_2h]) if actual_2h < total_2h else 0

    print(f"max 2-hit TP: {sorted_2h_vals[0]}, boundary_2h: {boundary_2h}")

    top_2h = sorted_2h_indices[:actual_2h]  # (actual_2h, 2) gene indices

    # Build 2-hit data matrices
    data_2h_tumor = tumor_gpu[top_2h[:, 0]] * tumor_gpu[top_2h[:, 1]]  # (top_k_2h, n_tumor)
    data_2h_normal = normal_gpu[top_2h[:, 0]] * normal_gpu[top_2h[:, 1]]  # (top_k_2h, n_normal)

    # Free 2-hit sorting arrays
    del i_idx, j_idx, upper_2h, sort_2h, sorted_2h_vals, sorted_2h_indices

    # === Step 2: 4-hit from 2-hit top-k ===
    mtx_4h_tp = data_2h_tumor @ data_2h_tumor.T  # (top_k_2h, top_k_2h)
    ri_4h, rj_4h = cp.triu_indices(actual_2h, k=1)
    upper_4h = mtx_4h_tp[ri_4h, rj_4h]
    del mtx_4h_tp

    # Filter for 4 distinct genes
    genes_4 = cp.stack([
        top_2h[ri_4h, 0], top_2h[ri_4h, 1],
        top_2h[rj_4h, 0], top_2h[rj_4h, 1]
    ], axis=1)
    genes_4_sorted = cp.sort(genes_4, axis=1)
    valid_4h = cp.all(cp.diff(genes_4_sorted, axis=1) != 0, axis=1)
    del genes_4, genes_4_sorted

    # Sort valid 4-hit by TP descending
    # Use -1 sentinel for invalid entries (TP values are >= 0)
    upper_4h_masked = cp.where(valid_4h, upper_4h, cp.int16(-1))
    sort_4h = cp.argsort(upper_4h_masked)[::-1]
    del upper_4h_masked

    n_valid_4h = int(cp.sum(valid_4h))
    del valid_4h
    if n_valid_4h == 0:
        print("No valid 4-hit entries, stopping.")
        break

    # Take only valid entries (first n_valid_4h after descending sort)
    sorted_4h_pos = sort_4h[:n_valid_4h]
    del sort_4h
    sorted_4h_tp = upper_4h[sorted_4h_pos]
    sorted_4h_ri = ri_4h[sorted_4h_pos]  # indices into top_2h
    sorted_4h_rj = rj_4h[sorted_4h_pos]
    del ri_4h, rj_4h, upper_4h, sorted_4h_pos

    # Gene indices for sorted 4-hit entries
    sorted_4h_genes = cp.stack([
        top_2h[sorted_4h_ri, 0], top_2h[sorted_4h_ri, 1],
        top_2h[sorted_4h_rj, 0], top_2h[sorted_4h_rj, 1]
    ], axis=1)  # (n_valid_4h, 4)

    print(f"Valid 4-hit entries: {n_valid_4h}, max 4-hit TP: {sorted_4h_tp[0]}")

    # === Step 3: 6-hit with chunked 4-hit expansion ===
    best_val = None
    best_genes = None

    n_chunk = 0
    while True:
        cs = n_chunk * top_k_4h
        ce = min((n_chunk + 1) * top_k_4h, n_valid_4h)

        if cs >= n_valid_4h:
            print("  4-hit entries exhausted.")
            break

        boundary_4h = int(sorted_4h_tp[ce]) if ce < n_valid_4h else 0

        # Build 4-hit data for this chunk
        c_ri = sorted_4h_ri[cs:ce]
        c_rj = sorted_4h_rj[cs:ce]
        c_4h_tumor = data_2h_tumor[c_ri] * data_2h_tumor[c_rj]  # (chunk_size, n_tumor)
        c_4h_normal = data_2h_normal[c_ri] * data_2h_normal[c_rj]  # (chunk_size, n_normal)
        c_4h_genes = sorted_4h_genes[cs:ce]  # (chunk_size, 4)

        # 6-hit score matrix: 4-hit chunk × all 2-hit (not symmetric)
        mtx_6h_tumor = c_4h_tumor @ data_2h_tumor.T  # (chunk_size, top_k_2h)
        mtx_6h_normal = c_4h_normal @ data_2h_normal.T
        mtx_6h = mtx_6h_tumor - 10 * mtx_6h_normal
        del mtx_6h_tumor, mtx_6h_normal

        result = find_best_valid_6h(mtx_6h, c_4h_genes, top_2h)
        del mtx_6h, c_4h_tumor, c_4h_normal, c_4h_genes, c_ri, c_rj
        if result and (best_val is None or result[0] > best_val):
            best_val = result[0]
            best_genes = list(result[1:])

        print(f"  4h-chunk {n_chunk}: [{cs}:{ce}], boundary_4h={boundary_4h}, best_val={best_val}")

        # Stopping condition: remaining 4-hit entries can't improve
        if best_val is not None and boundary_4h <= best_val:
            print(f"  4-hit expansion done (boundary_4h={boundary_4h} <= best_val={best_val}).")
            break

        n_chunk += 1
    
    # Check global optimality after chunk loop
    if best_val is not None and best_val >= boundary_2h:
        print(f"  best_val ({best_val}) >= boundary_2h ({boundary_2h}), globally optimal.")
    elif best_val is not None:
        print(f"  WARNING: best_val ({best_val}) < boundary_2h ({boundary_2h}), may need 2-hit expansion.")

    # Clean up 4-hit arrays
    del sorted_4h_tp, sorted_4h_ri, sorted_4h_rj, sorted_4h_genes
    del data_2h_tumor, data_2h_normal

    if best_genes is not None:
        idx_a, idx_b, idx_c, idx_d, idx_e, idx_f = best_genes
        mask = (tumor_gpu[idx_a] & tumor_gpu[idx_b] & tumor_gpu[idx_c] &
                tumor_gpu[idx_d] & tumor_gpu[idx_e] & tumor_gpu[idx_f]).astype(bool)
        removed = int(cp.sum(mask))
        mask_normal = (normal_gpu[idx_a] & normal_gpu[idx_b] & normal_gpu[idx_c] &
                       normal_gpu[idx_d] & normal_gpu[idx_e] & normal_gpu[idx_f]).astype(bool)
        covered_normals = int(cp.sum(mask_normal))
        if removed == 0:
            print(f"No tumors removed with genes {best_genes}, stopping.")
            break
        print(f"Removing {removed} tumors, {covered_normals} normals covered, genes {best_genes}")
        results.append({
            'iter': iteration,
            'genes': [int(g) for g in best_genes],
            'score': best_val,
            'removed': removed,
            'normals_covered': covered_normals,
        })
        tumor_gpu = tumor_gpu[:, ~mask]
    else:
        print("No valid 6-hit combination found, stopping.")
        break

print(f"\nFinal tumor matrix shape: {tumor_gpu.shape}")
print(f"Total loop time: {time.time() - total_start:.2f}s")
print_result_genes(results)
