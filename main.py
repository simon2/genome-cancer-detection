import sys

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python main.py <data_file> <hits>")
        print(f"  data_file: path to data file (e.g., data/BLCA.txt)")
        print(f"  hits: number of hits (2, 3, 4, 6, 8, 9)")
        sys.exit(1)

    data_file = sys.argv[1]
    hits = int(sys.argv[2])

    if hits == 2:
        from test_2h import run
    elif hits == 3:
        from test_3h import run
    elif hits == 4:
        from test_4h import run
    elif hits == 5:
        from test_5h import run
    elif hits == 6:
        from test_6h import run
    elif hits == 7:
        from test_7h import run
    elif hits == 8:
        from test_8h import run
    elif hits == 9:
        from test_9h import run
    else:
        print(f"Unsupported hit number: {hits}. Supported: 2, 3, 4, 5, 6, 7, 8, 9")
        sys.exit(1)

    run(data_file)

if __name__ == '__main__':
    main()
