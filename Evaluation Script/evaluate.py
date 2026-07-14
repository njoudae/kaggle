import os
import sys
import zipfile
import pandas as pd
from sklearn.metrics import f1_score

VALID_LABELS = ["Against", "Favor", "None"]
LABEL2ID = {"Against": 0, "Favor": 1, "None": 2}


def find_file(folder, extensions):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(extensions):
                return os.path.join(root, f)
    return None


def safe_name(name):
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def load_gold(ref_dir):
    gold_path = find_file(ref_dir, (".csv",))
    if gold_path is None:
        raise FileNotFoundError("No gold CSV file found in reference folder.")

    gold = pd.read_csv(gold_path, keep_default_na=False)
    gold.columns = gold.columns.astype(str).str.strip()

    required_cols = ["target", "stance"]
    for col in required_cols:
        if col not in gold.columns:
            raise ValueError(f"Gold file must contain a '{col}' column.")

    gold["target"] = gold["target"].astype(str).str.strip()
    gold["stance"] = gold["stance"].astype(str).str.strip()

    invalid = sorted(set(gold["stance"]) - set(VALID_LABELS))
    if invalid:
        raise ValueError(f"Invalid labels in gold file: {invalid}")

    return gold


def load_predictions(res_dir):
    zip_path = find_file(res_dir, (".zip",))

    if zip_path:
        extract_dir = os.path.join(res_dir, "extracted_submission")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        pred_path = find_file(extract_dir, (".txt",))
    else:
        pred_path = find_file(res_dir, (".txt",))

    if pred_path is None:
        raise FileNotFoundError("No prediction TXT file found.")

    with open(pred_path, "r", encoding="utf-8") as f:
        preds = [line.strip() for line in f if line.strip() != ""]

    if preds and preds[0].lower() == "stance":
        preds = preds[1:]

    invalid = sorted(set(preds) - set(VALID_LABELS))
    if invalid:
        raise ValueError(f"Invalid prediction labels: {invalid}")

    return preds


def compute_metrics(gold_labels, pred_labels):
    y_true = [LABEL2ID[x] for x in gold_labels]
    y_pred = [LABEL2ID[x] for x in pred_labels]

    f_against = f1_score(y_true, y_pred, labels=[0], average="macro")
    f_favor = f1_score(y_true, y_pred, labels=[1], average="macro")
    f_none = f1_score(y_true, y_pred, labels=[2], average="macro")

    favg2 = (f_favor + f_against) / 2.0
    favg3 = (f_favor + f_against + f_none) / 3.0

    return {
        "Favg2": favg2,
        "Favg3": favg3,
    }


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    ref_dir = os.path.join(input_dir, "ref")
    res_dir = os.path.join(input_dir, "res")

    os.makedirs(output_dir, exist_ok=True)

    gold_df = load_gold(ref_dir)
    preds = load_predictions(res_dir)

    if len(preds) != len(gold_df):
        raise ValueError(
            f"Length mismatch: expected {len(gold_df)} predictions, got {len(preds)}."
        )

    gold_df["pred"] = preds

    overall_metrics = compute_metrics(
        gold_df["stance"].tolist(),
        gold_df["pred"].tolist()
    )

    scores_path = os.path.join(output_dir, "scores.txt")

    with open(scores_path, "w", encoding="utf-8") as f:
        for target in sorted(gold_df["target"].unique()):
            sub = gold_df[gold_df["target"] == target]

            m = compute_metrics(
                sub["stance"].tolist(),
                sub["pred"].tolist()
            )

            target_name = safe_name(target)

            f.write(f"{target_name}_Favg2={m['Favg2']:.6f}\n")
            f.write(f"{target_name}_Favg3={m['Favg3']:.6f}\n")

        f.write(f"Overall_Favg2={overall_metrics['Favg2']:.6f}\n")
        f.write(f"Overall_Favg3={overall_metrics['Favg3']:.6f}\n")


if __name__ == "__main__":
    main()
