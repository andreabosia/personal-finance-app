# backend/cli/classify_csv.py
import argparse, json
from backend.services.classifier_api.bootstrap import classify_csv_idempotent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--id_col", default="transaction_id")
    p.add_argument("--merchant_col", default="merchant")
    args = p.parse_args()
    out = classify_csv_idempotent(args.csv_path, args.models, args.id_col, args.merchant_col)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()


# e.g run from command line:
# python -m backend.cli.classify_csv --csv_path data/transactions.csv --models embedding zero_shot