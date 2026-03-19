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


tumor, normal = read_data('data/BLCA.txt')
print(f"Tumor matrix shape:     {tumor.shape}")
print(f"Non-tumor matrix shape: {normal.shape}")

def print_results(results):
    print("\n=== Summary ===")
    for r in results:
        print(f"Iter {r['iter']}: genes ({r['genes'][0]},{r['genes'][1]})  "
              f"score={r['score']}  removed={r['removed']}  normals_covered={r['normals_covered']}")

def print_result_genes(results):
    print("\n=== Genes ===")
    for r in results:
        print(f"{r['genes'][0]},{r['genes'][1]}")

iteration = 0
results = []

tumor_gpu = cp.asarray(tumor, dtype=cp.float16)
normal_gpu = cp.asarray(normal, dtype=cp.float16)

total_start = time.time()

while tumor_gpu.shape[1] > 0:
    iteration += 1
    print(f"\n=== Iteration {iteration} ===")
    print(f"Tumor matrix shape: {tumor_gpu.shape}")

    # Score matrix: TP - 10 * FP
    mtx_tp = tumor_gpu @ tumor_gpu.T
    mtx_fp = normal_gpu @ normal_gpu.T
    mtx_score = mtx_tp - 10 * mtx_fp

    # Find max in upper triangle (i < j)
    i_idx, j_idx = cp.triu_indices(mtx_score.shape[0], k=1)
    upper_scores = mtx_score[i_idx, j_idx]
    flat_idx = int(cp.argmax(upper_scores))
    best_val = int(upper_scores[flat_idx])
    best_i = int(i_idx[flat_idx])
    best_j = int(j_idx[flat_idx])

    print(f"Best score: {best_val}, genes: ({best_i}, {best_j})")

    # Remove covered tumors
    mask_tumor = (tumor_gpu[best_i] * tumor_gpu[best_j]).astype(bool)
    removed = int(cp.sum(mask_tumor))
    mask_normal = (normal_gpu[best_i] * normal_gpu[best_j]).astype(bool)
    covered = int(cp.sum(mask_normal))

    if removed == 0:
        print(f"No tumors removed with genes {best_i},{best_j}, stopping.")
        break

    print(f"Removing {removed} tumors ({covered} normals covered) where genes {best_i},{best_j} are all 1")
    results.append({
        'iter': iteration,
        'genes': [best_i, best_j],
        'score': best_val,
        'removed': removed,
        'normals_covered': covered,
    })
    tumor_gpu = tumor_gpu[:, ~mask_tumor]

    del mtx_tp, mtx_fp, mtx_score, i_idx, j_idx, upper_scores

print(f"\nFinal tumor matrix shape: {tumor_gpu.shape}")
print(f"Total loop time: {time.time() - total_start:.2f}s")
# print_results(results)
print_result_genes(results)
