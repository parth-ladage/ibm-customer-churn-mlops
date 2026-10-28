import os
import csv
import random

def split_data():
    raw_path = "data/customer_churn.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not found.")

    print(f"Loading real dataset from: {raw_path}")
    
    with open(raw_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    total_records = len(all_rows)
    print(f"Total records in real dataset: {total_records}")

    # Stratified split based on 'Churn' column ("Yes" vs "No")
    churn_yes = [r for r in all_rows if r.get("Churn") == "Yes"]
    churn_no = [r for r in all_rows if r.get("Churn") == "No"]

    print(f"Real class distribution: Churn=Yes ({len(churn_yes)}), Churn=No ({len(churn_no)})")

    # Set seed for reproducible split
    random.seed(42)
    random.shuffle(churn_yes)
    random.shuffle(churn_no)

    split_ratio = 0.80

    split_idx_yes = int(len(churn_yes) * split_ratio)
    split_idx_no = int(len(churn_no) * split_ratio)

    train_rows = churn_yes[:split_idx_yes] + churn_no[:split_idx_no]
    test_rows = churn_yes[split_idx_yes:] + churn_no[split_idx_no:]

    random.shuffle(train_rows)
    random.shuffle(test_rows)

    fieldnames = all_rows[0].keys()

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    with open(train_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    with open(test_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"Successfully performed 80/20 stratified split on real data:")
    print(f" - Train set saved to: {train_path} ({len(train_rows)} rows)")
    print(f" - Test set saved to:  {test_path} ({len(test_rows)} rows)")

if __name__ == "__main__":
    split_data()

