import os
import json
import pandas as pd

# Create output directory
output_dir = "cleaned_results"
os.makedirs(output_dir, exist_ok=True)

alphas = ["0.2", "0.4", "0.6", "0.8"]
opponent_types = ["aggressive", "loose", "passive", "tight"]

for opponent in opponent_types:
    rows = []

    for alpha in alphas:
        json_path = f"results/{opponent}_opponents_{alpha}_summary_alpha_{alpha}.json"
        if not os.path.exists(json_path):
            print(f"Missing: {json_path}")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        key = f"{opponent}-alpha-{alpha}"
        row = {
            "Alpha": float(alpha),
            "Hands": data["sim_params"]["hands"],
            "Trials": data["sim_params"]["trials"],
            "Adaptive Bankroll": data["adaptive_perf"].get(key, None),
            "Nash Bankroll": data["nash_perf"].get(key, None),
            "KL Divergence": data["adaptive_kl"].get(key, None),
        }
        rows.append(row)

    # Convert to DataFrame and sort
    df = pd.DataFrame(rows)
    df = df.sort_values(by="Alpha")

    # Save as CSV
    csv_path = os.path.join(output_dir, f"{opponent}_opponents_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary for {opponent}: {csv_path}")
