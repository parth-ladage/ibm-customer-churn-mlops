import os
import csv

def prepare_reference_and_current():
    iris_path = "data/iris.csv"
    if not os.path.exists(iris_path):
        raise FileNotFoundError(f"{iris_path} not found.")

    with open(iris_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Loaded real Iris dataset: {len(all_rows)} rows.")

    # Split real Iris observations into Reference baseline (first half) and Current production batch (second half)
    # Stratified by species
    setosa = [r for r in all_rows if r["species"] == "setosa"]
    versicolor = [r for r in all_rows if r["species"] == "versicolor"]
    virginica = [r for r in all_rows if r["species"] == "virginica"]

    ref_rows = setosa[:25] + versicolor[:25] + virginica[:25]
    curr_rows = setosa[25:] + versicolor[25:] + virginica[25:]

    ref_path = "data/iris_reference.csv"
    curr_path = "data/iris_current.csv"

    fieldnames = all_rows[0].keys()

    with open(ref_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ref_rows)

    with open(curr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curr_rows)

    print(f"Reference dataset created: {ref_path} ({len(ref_rows)} rows)")
    print(f"Current dataset created:   {curr_path} ({len(curr_rows)} rows)")

if __name__ == "__main__":
    prepare_reference_and_current()

