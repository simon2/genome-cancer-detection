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


def find_best_valid_3h(mtx_3h, genes_2h, n_genes):
    """Find best valid 3-hit combo from score matrix.
    mtx_3h: (n_2h, n_genes) score matrix on GPU, NOT symmetric
    genes_2h: (n_2h, 2) gene indices on GPU (row side)
    n_genes: total number of genes (column index = gene index)
    Returns (max_val, g1, g2, g3) or None.
    """
    n_2h, n_g = mtx_3h.shape

    # Check 3 distinct genes: the column index (3rd gene) must not equal either of the 2 row genes
    gene_col = cp.arange(n_g, dtype=cp.int32)[None, :]  # (1, n_genes)
    overlap = cp.zeros((n_2h, n_g), dtype=bool)
    for k in range(2):
        overlap |= (genes_2h[:, k:k+1] == gene_col)
    valid = ~overlap

    if not cp.any(valid):
        return None

    mtx_masked = cp.where(valid, mtx_3h, cp.iinfo(mtx_3h.dtype).min)
    flat_idx = int(cp.argmax(mtx_masked))
    p = flat_idx // n_g
    q = flat_idx % n_g
    max_val = int(mtx_3h[p, q])

    genes_out = list(cp.asnumpy(genes_2h[p])) + [q]
    return (max_val, *genes_out)


tumor, normal = read_data('data/HNSC.txt')
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

top_k_2h = 10000
iteration = 0
results = []

tumor_gpu = cp.asarray(tumor, dtype=cp.int16)
normal_gpu = cp.asarray(normal, dtype=cp.int16)

total_start = time.time()

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

    # === Step 2: 3-hit with chunked 2-hit expansion ===
    # 3-hit = 2-hit chunk × all original genes (not symmetric, column side is fixed)
    best_val = None
    best_genes = None

    n_chunk = 0
    while True:
        cs = n_chunk * top_k_2h
        ce = min((n_chunk + 1) * top_k_2h, actual_2h)

        if cs >= actual_2h:
            print("  2-hit entries exhausted.")
            break

        # 3-hit score matrix: 2-hit chunk × all genes
        c_2h_tumor = data_2h_tumor[cs:ce]  # (chunk_size, n_tumor)
        c_2h_normal = data_2h_normal[cs:ce]  # (chunk_size, n_normal)
        c_2h_genes = top_2h[cs:ce]  # (chunk_size, 2)

        mtx_3h_tumor = c_2h_tumor @ tumor_gpu.T  # (chunk_size, n_genes)
        mtx_3h_normal = c_2h_normal @ normal_gpu.T  # (chunk_size, n_genes)
        mtx_3h = mtx_3h_tumor - 10 * mtx_3h_normal
        del mtx_3h_tumor, mtx_3h_normal

        result = find_best_valid_3h(mtx_3h, c_2h_genes, tumor_gpu.shape[0])
        del mtx_3h, c_2h_tumor, c_2h_normal, c_2h_genes

        if result and (best_val is None or result[0] > best_val):
            best_val = result[0]
            best_genes = list(result[1:])

        print(f"  2h-chunk {n_chunk}: [{cs}:{ce}], boundary_2h={boundary_2h}, best_val={best_val}")

        # Stopping condition: remaining 2-hit entries can't improve
        if best_val is not None and boundary_2h <= best_val:
            print(f"  2-hit expansion done (boundary_2h={boundary_2h} <= best_val={best_val}).")
            break

        n_chunk += 1

    # Global optimality check
    if best_val is not None and best_val >= boundary_2h:
        print(f"  best_val ({best_val}) >= boundary_2h ({boundary_2h}), globally optimal.")
    else:
        print(f"  !!!!! WARNING: best_val ({best_val}) < boundary_2h ({boundary_2h}), result may NOT be globally optimal! Need 2-hit expansion! !!!!!")

    # Clean up
    del data_2h_tumor, data_2h_normal

    if best_genes is not None:
        idx_a, idx_b, idx_c = best_genes
        mask = (tumor_gpu[idx_a] & tumor_gpu[idx_b] & tumor_gpu[idx_c]).astype(bool)
        removed = int(cp.sum(mask))
        mask_normal = (normal_gpu[idx_a] & normal_gpu[idx_b] & normal_gpu[idx_c]).astype(bool)
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
        print("No valid 3-hit combination found, stopping.")
        break

print(f"\nFinal tumor matrix shape: {tumor_gpu.shape}")
print(f"Total loop time: {time.time() - total_start:.2f}s")
print_result_genes(results)
