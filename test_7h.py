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


def extract_top_valid_6h(mtx_tp, genes_4h, genes_2h, top_k):
    """Extract top-k valid 6-hit entries (by TP) from a TP matrix.
    mtx_tp: (n_4h, n_2h) TP-only matrix on GPU, NOT symmetric
    genes_4h: (n_4h, 4) gene indices on GPU (row side)
    genes_2h: (n_2h, 2) gene indices on GPU (col side)
    Returns (tps, genes) arrays or (None, None).
    """
    n_4h, n_2h = mtx_tp.shape

    # Check 6 distinct genes (4 from row × 2 from col)
    overlap = cp.zeros((n_4h, n_2h), dtype=bool)
    for k in range(4):
        for l in range(2):
            overlap |= (genes_4h[:, k:k+1] == genes_2h[None, :, l])
    valid = ~overlap
    del overlap

    n_valid = int(cp.sum(valid))
    if n_valid == 0:
        del valid
        return None, None

    mtx_masked = cp.where(valid, mtx_tp, -1)
    del valid
    flat = mtx_masked.ravel()
    k_actual = min(top_k, n_valid)
    top_flat_idx = cp.argpartition(flat, -k_actual)[-k_actual:]
    top_tp = flat[top_flat_idx]
    del flat, mtx_masked

    keep = top_tp >= 0
    top_flat_idx = top_flat_idx[keep]
    top_tp = top_tp[keep]

    if len(top_tp) == 0:
        return None, None

    rows = top_flat_idx // n_2h
    cols = top_flat_idx % n_2h
    top_genes = cp.concatenate([genes_4h[rows], genes_2h[cols]], axis=1)  # (k, 6)
    return top_tp, top_genes


def find_best_valid_7h(mtx_7h, genes_6h):
    """Find best valid 7-hit combo from score matrix.
    mtx_7h: (n_6h, n_genes) score matrix on GPU
    genes_6h: (n_6h, 6) gene indices on GPU (row side)
    Column index j = gene index j.
    Returns (max_val, g1..g7) or None.
    """
    n_6h, n_genes = mtx_7h.shape

    # Check that the 7th gene (column) is not in the 6 genes (row)
    gene_col = cp.arange(n_genes, dtype=cp.int32)
    overlap = cp.zeros((n_6h, n_genes), dtype=bool)
    for k in range(6):
        overlap |= (genes_6h[:, k:k+1] == gene_col[None, :])
    valid = ~overlap
    del overlap

    if not cp.any(valid):
        return None

    mtx_masked = cp.where(valid, mtx_7h, cp.finfo(mtx_7h.dtype).min)
    del valid
    flat_idx = int(cp.argmax(mtx_masked))
    del mtx_masked
    p = flat_idx // n_genes
    q = flat_idx % n_genes
    max_val = int(mtx_7h[p, q])

    genes_out = list(cp.asnumpy(genes_6h[p])) + [int(q)]
    return (max_val, *genes_out)


def print_result_genes(results):
    print("\n=== Genes ===")
    for r in results:
        print(','.join(str(g) for g in r['genes']))

def run(data_file):
    tumor, normal = read_data(data_file)
    print(f"Tumor matrix shape:     {tumor.shape}")
    print(f"Non-tumor matrix shape: {normal.shape}")

    top_k_2h = 25000
    top_k_4h = 50000
    top_k_6h = 10000
    iteration = 0
    results = []

    tumor_gpu = cp.asarray(tumor, dtype=cp.float16)
    normal_gpu = cp.asarray(normal, dtype=cp.float16)

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

        top_2h = sorted_2h_indices[:actual_2h]

        # Build 2-hit data matrices
        data_2h_tumor = tumor_gpu[top_2h[:, 0]] * tumor_gpu[top_2h[:, 1]]
        data_2h_normal = normal_gpu[top_2h[:, 0]] * normal_gpu[top_2h[:, 1]]

        del i_idx, j_idx, upper_2h, sort_2h, sorted_2h_vals, sorted_2h_indices

        # === Step 2: 4-hit from 2-hit top-k ===
        mtx_4h_tp = data_2h_tumor @ data_2h_tumor.T
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
        upper_4h_masked = cp.where(valid_4h, upper_4h, cp.int16(-1))
        sort_4h = cp.argsort(upper_4h_masked)[::-1]
        del upper_4h_masked

        n_valid_4h = int(cp.sum(valid_4h))
        del valid_4h
        if n_valid_4h == 0:
            print("No valid 4-hit entries, stopping.")
            break

        sorted_4h_pos = sort_4h[:n_valid_4h]
        del sort_4h
        sorted_4h_tp = upper_4h[sorted_4h_pos]
        sorted_4h_ri = ri_4h[sorted_4h_pos]
        sorted_4h_rj = rj_4h[sorted_4h_pos]
        del ri_4h, rj_4h, upper_4h, sorted_4h_pos

        sorted_4h_genes = cp.stack([
            top_2h[sorted_4h_ri, 0], top_2h[sorted_4h_ri, 1],
            top_2h[sorted_4h_rj, 0], top_2h[sorted_4h_rj, 1]
        ], axis=1)

        print(f"Valid 4-hit entries: {n_valid_4h}, max 4-hit TP: {sorted_4h_tp[0]}")

        # === Step 3+4: Nested loops — outer expands 4-hit, inner searches 7-hit ===
        best_val = None
        best_genes = None

        boundary_4h = 0
        n_chunk_4 = 0

        # Outer loop: expand 4-hit chunks, collect NEW 6-hit entries, search for 7-hit
        while True:
            cs = n_chunk_4 * top_k_4h
            ce = min((n_chunk_4 + 1) * top_k_4h, n_valid_4h)

            if cs >= n_valid_4h:
                boundary_4h = 0
                print("  4-hit entries exhausted.")
                break

            boundary_4h = int(sorted_4h_tp[ce]) if ce < n_valid_4h else 0

            # Build 4-hit data for this chunk
            c_ri = sorted_4h_ri[cs:ce]
            c_rj = sorted_4h_rj[cs:ce]
            c_4h_tumor = data_2h_tumor[c_ri] * data_2h_tumor[c_rj]
            c_4h_normal = data_2h_normal[c_ri] * data_2h_normal[c_rj]
            c_4h_genes = sorted_4h_genes[cs:ce]

            # Collect NEW 6-hit entries: 4-hit chunk × all 2-hit (not symmetric)
            mtx_6h_tp = c_4h_tumor @ data_2h_tumor.T  # (chunk_size, top_k_2h)
            tp, genes = extract_top_valid_6h(mtx_6h_tp, c_4h_genes, top_2h, top_k_6h)
            del mtx_6h_tp

            n_new = len(tp) if tp is not None else 0
            print(f"  4h-chunk {n_chunk_4}: [{cs}:{ce}], boundary_4h={boundary_4h}, new 6h entries={n_new}")

            if n_new == 0:
                del c_4h_tumor, c_4h_normal, c_4h_genes, c_ri, c_rj
                n_chunk_4 += 1
                continue

            # Sort NEW 6-hit entries by TP descending
            sort_new = cp.argsort(tp)[::-1]
            actual_new = min(top_k_6h, len(tp))
            sorted_new_tp = tp[sort_new[:actual_new]]
            sorted_new_genes = genes[sort_new[:actual_new]]
            del tp, genes, sort_new

            print(f"  New 6-hit entries sorted: {actual_new}, max 6-hit TP: {sorted_new_tp[0]}")

            # Inner loop: chunk through NEW 6-hit entries to find best 7-hit
            n_chunk_6 = 0
            while True:
                cs6 = n_chunk_6 * top_k_6h
                ce6 = min((n_chunk_6 + 1) * top_k_6h, actual_new)

                if cs6 >= actual_new:
                    print("    New 6-hit entries exhausted.")
                    break

                boundary_6h = int(sorted_new_tp[ce6]) if ce6 < actual_new else 0

                # Build 6-hit data for this chunk from gene indices
                c_6h_genes = sorted_new_genes[cs6:ce6]
                chunk_size = ce6 - cs6

                c_6h_tumor = cp.ones((chunk_size, tumor_gpu.shape[1]), dtype=cp.float16)
                for k in range(6):
                    c_6h_tumor *= tumor_gpu[c_6h_genes[:, k]]
                c_6h_normal = cp.ones((chunk_size, normal_gpu.shape[1]), dtype=cp.float16)
                for k in range(6):
                    c_6h_normal *= normal_gpu[c_6h_genes[:, k]]

                # 7-hit score matrix: 6-hit chunk × all genes (not symmetric)
                mtx_7h_tumor = c_6h_tumor @ tumor_gpu.T
                mtx_7h_normal = c_6h_normal @ normal_gpu.T
                mtx_7h = mtx_7h_tumor - 10 * mtx_7h_normal
                del mtx_7h_tumor, mtx_7h_normal, c_6h_tumor, c_6h_normal

                result = find_best_valid_7h(mtx_7h, c_6h_genes)
                del mtx_7h
                if result and (best_val is None or result[0] > best_val):
                    best_val = result[0]
                    best_genes = list(result[1:])

                print(f"    6h-chunk {n_chunk_6}: [{cs6}:{ce6}], boundary_6h={boundary_6h}, best_val={best_val}")

                # Inner stopping: remaining new 6-hit entries can't improve
                if best_val is not None and boundary_6h <= best_val:
                    print(f"    6-hit expansion done (boundary_6h={boundary_6h} <= best_val={best_val}).")
                    break

                n_chunk_6 += 1

            del sorted_new_tp, sorted_new_genes
            del c_4h_tumor, c_4h_normal, c_4h_genes, c_ri, c_rj

            # Outer stopping: check if best_val >= boundary_4h
            if best_val is not None and boundary_4h <= best_val:
                print(f"  4-hit expansion done (boundary_4h={boundary_4h} <= best_val={best_val}).")
                break

            n_chunk_4 += 1

        del sorted_4h_tp, sorted_4h_ri, sorted_4h_rj, sorted_4h_genes
        del data_2h_tumor, data_2h_normal

        # Check global optimality
        if best_val is not None and best_val >= boundary_2h:
            print(f"  best_val ({best_val}) >= boundary_2h ({boundary_2h}), globally optimal.")
        elif best_val is not None:
            print(f"  WARNING: best_val ({best_val}) < boundary_2h ({boundary_2h}), may need 2-hit expansion.")

        if best_genes is not None:
            mask = tumor_gpu[best_genes[0]].astype(bool)
            for k in range(1, 7):
                mask *= tumor_gpu[best_genes[k]].astype(bool)
            removed = int(cp.sum(mask))

            mask_normal = normal_gpu[best_genes[0]].astype(bool)
            for k in range(1, 7):
                mask_normal *= normal_gpu[best_genes[k]].astype(bool)
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
            print("No valid 7-hit combination found, stopping.")
            break

    print(f"\nFinal tumor matrix shape: {tumor_gpu.shape}")
    print(f"Total loop time: {time.time() - total_start:.2f}s")
    print_result_genes(results)
    return results

if __name__ == '__main__':
    run('data/BLCA.txt')
