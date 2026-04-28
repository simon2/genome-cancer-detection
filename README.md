# Genome Cancer Detection

GPU-accelerated gene combination pruning tool for cancer detection. Given a binary gene expression matrix (tumor vs. non-tumor samples), it finds minimal sets of genes (k-hit combinations) that can distinguish tumor samples from non-tumor samples.

## Requirements

- Python 3
- NumPy
- CuPy (CUDA GPU required)

## Usage

Run a k-hit test on a data file:

```bash
python main.py <data_file> <hits>
```

- `data_file` — path to the input data file (e.g., `data/BLCA.txt`)
- `hits` — number of gene hits to test (2–9)

Example:

```bash
python main.py data/BLCA.txt 4
```

### Verifying Results

```bash
python test_result.py <test_data.txt> <result.txt>
```

## Supported Hit Counts

| Hits | Script |
|------|-------------|
| 2 | `test_2h.py` |
| 3 | `test_3h.py` |
| 4 | `test_4h.py` |
| 5 | `test_5h.py` |
| 6 | `test_6h.py` |
| 7 | `test_7h.py` |
| 8 | `test_8h.py` |
| 9 | `test_9h.py` |

## Input Data Format

A text file where the first line is a header containing: number of genes, total samples, -1 (unused), number of tumor samples, and number of non-tumor samples. Each subsequent line is a binary row (one character per sample: `0` or `1`) representing gene expression, with tumor samples first followed by non-tumor samples.

Example (6 genes, 7 samples total, 4 tumor + 3 non-tumor):

```
6 7 -1 4 3
1101000
0010110
1110001
0100100
0011010
1001001
```

The first 4 characters of each row are tumor samples and the last 3 are non-tumor samples.
