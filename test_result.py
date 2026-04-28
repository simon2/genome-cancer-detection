import numpy as np
import sys

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
            row = np.frombuffer(line.encode(), dtype=np.uint8) - ord('0')
            tumor_matrix[i] = row[:n_tumor]
            non_tumor_matrix[i] = row[n_tumor:]

    return tumor_matrix, non_tumor_matrix, n_tumor, n_non_tumor

def test_result(data_file, result_file):
    tumor, normal, n_tumor, n_normal = read_data(data_file)
    print(f"Test data: {n_tumor} tumor, {n_normal} normal")

    # Track which samples are still alive (not covered)
    tumor_alive = np.ones(n_tumor, dtype=bool)
    normal_alive = np.ones(n_normal, dtype=bool)

    with open(result_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            genes = [int(x) for x in line.strip().split(',')]

            # Check tumor: all genes must be 1 for a sample to be covered
            tumor_match = np.ones(n_tumor, dtype=bool)
            for g in genes:
                tumor_match &= tumor[g].astype(bool)
            # Only cover samples still alive
            tumor_covered_now = tumor_match & tumor_alive
            tumor_alive[tumor_covered_now] = False

            # Check normal
            normal_match = np.ones(n_normal, dtype=bool)
            for g in genes:
                normal_match &= normal[g].astype(bool)
            normal_covered_now = normal_match & normal_alive
            normal_alive[normal_covered_now] = False

            t_covered = np.sum(tumor_covered_now)
            n_covered = np.sum(normal_covered_now)
            if t_covered > 0 or n_covered > 0:
                print(f"Rule {line_num} {genes}: dropped {t_covered} tumor, {n_covered} normal")

    total_tumor_covered = n_tumor - np.sum(tumor_alive)
    total_normal_covered = n_normal - np.sum(normal_alive)
    print(f"\nTotal tumor covered:  {total_tumor_covered}/{n_tumor} ({100*total_tumor_covered/n_tumor:.1f}%)")
    print(f"Total normal covered: {total_normal_covered}/{n_normal} ({100*total_normal_covered/n_normal:.1f}%)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <test_data.txt> <result.txt>")
        sys.exit(1)
    test_result(sys.argv[1], sys.argv[2])
