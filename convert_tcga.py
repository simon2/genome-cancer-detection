import numpy as np
import sys
import os

DATA_DIR = 'data/gen'


def find_csv(name, kind):
    # CSVs live either in data/gen/<NAME>/ or flat in data/gen/
    for path in (f'{DATA_DIR}/{name}/{name}-{kind}.csv', f'{DATA_DIR}/{name}-{kind}.csv'):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'{name}-{kind}.csv not found under {DATA_DIR}/')


def convert(name, seed=42):
    out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Read tumor
    with open(find_csv(name, 'tumor')) as f:
        header = f.readline().strip().split(',')
        n_genes_t, n_tumor = int(header[0]), int(header[1])
        tumor_data = np.empty((n_genes_t, n_tumor), dtype=np.uint8)
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            # parts[0] = id, parts[1] = gene name, parts[2:] = data
            tumor_data[i] = np.array(parts[2:], dtype=np.uint8)

    # Read normal
    with open(find_csv(name, 'normal')) as f:
        header = f.readline().strip().split(',')
        n_genes_n, n_normal = int(header[0]), int(header[1])
        normal_data = np.empty((n_genes_n, n_normal), dtype=np.uint8)
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            normal_data[i] = np.array(parts[2:], dtype=np.uint8)

    assert n_genes_t == n_genes_n, f"Gene count mismatch: {n_genes_t} vs {n_genes_n}"
    n_genes = n_genes_t

    # Shuffle columns (samples) for train/test split
    rng = np.random.default_rng(seed)
    tumor_perm = rng.permutation(n_tumor)
    normal_perm = rng.permutation(n_normal)

    n_tumor_train = int(n_tumor * 0.75)
    n_normal_train = int(n_normal * 0.75)
    n_tumor_test = n_tumor - n_tumor_train
    n_normal_test = n_normal - n_normal_train

    tumor_train = tumor_data[:, tumor_perm[:n_tumor_train]]
    tumor_test = tumor_data[:, tumor_perm[n_tumor_train:]]
    normal_train = normal_data[:, normal_perm[:n_normal_train]]
    normal_test = normal_data[:, normal_perm[n_normal_train:]]

    def write_file(path, tumor_part, normal_part):
        n_t = tumor_part.shape[1]
        n_n = normal_part.shape[1]
        total = n_t + n_n
        # Concatenate tumor columns first, then normal
        combined = np.concatenate([tumor_part, normal_part], axis=1)
        with open(path, 'w') as f:
            f.write(f"{n_genes} {total} -1 {n_t} {n_n}\n")
            for row in combined:
                f.write(''.join(str(v) for v in row) + '\n')

    write_file(f'{out_dir}/{name}_train.txt', tumor_train, normal_train)
    write_file(f'{out_dir}/{name}_test.txt', tumor_test, normal_test)

    print(f"{name} train: {n_genes} genes, {n_tumor_train} tumor, {n_normal_train} normal")
    print(f"{name} test:  {n_genes} genes, {n_tumor_test} tumor, {n_normal_test} normal")


if __name__ == '__main__':
    convert('BRCA')
    convert('PRAD')
