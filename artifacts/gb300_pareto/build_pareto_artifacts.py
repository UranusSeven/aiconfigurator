from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parent
OUT_CSV = BASE / "gb300_sota_raw_pareto_results.csv"
OUT_FRONTIER_CSV = BASE / "gb300_sota_pareto_frontier.csv"
OUT_PNG = BASE / "gb300_sota_pareto_tps_vs_per_gpu.png"


RUN_LABELS = {
    "kimi": "Kimi K2.5 NVFP4",
    "minimax": "MiniMax M2.7 NVFP4",
    "glm_trtllm": "GLM-5 NVFP4",
    "glm_vllm": "GLM-5 NVFP4",
}


def clean_backend(row: pd.Series) -> str:
    backend = row.get("backend")
    if pd.notna(backend) and backend:
        return str(backend)
    prefill_backend = row.get("(p)backend")
    decode_backend = row.get("(d)backend")
    if pd.notna(prefill_backend) and pd.notna(decode_backend):
        if prefill_backend == decode_backend:
            return str(prefill_backend)
        return f"{prefill_backend}+{decode_backend}"
    if pd.notna(prefill_backend):
        return str(prefill_backend)
    if pd.notna(decode_backend):
        return str(decode_backend)
    return "unknown"


def source_run(path: Path) -> str:
    rel = path.relative_to(BASE)
    return rel.parts[0]


frames = []
for csv_path in sorted(BASE.glob("*/*/*/*/pareto.csv")):
    run = source_run(csv_path)
    mode = csv_path.parent.name
    df = pd.read_csv(csv_path)
    df.insert(0, "source_run", run)
    df.insert(1, "model_label", RUN_LABELS.get(run, run))
    df.insert(2, "serving_mode", mode)
    df.insert(3, "source_file", str(csv_path.relative_to(BASE)))
    frames.append(df)

if not frames:
    raise SystemExit(f"No pareto.csv files found below {BASE}")

raw = pd.concat(frames, ignore_index=True, sort=False)
raw["backend_label"] = raw.apply(clean_backend, axis=1)
raw["satisfies_sla"] = (raw["ttft"] <= 2000) & (raw["tpot"] <= 30)
raw["cluster_total_gpus"] = 72
raw["raw_candidate_tps"] = raw["tokens/s"]
raw["cluster_tps_72gpu"] = raw["tokens/s/gpu_cluster"] * raw["cluster_total_gpus"]
raw["per_gpu_throughput_72gpu"] = raw["tokens/s/gpu_cluster"]
raw["plot_label"] = (
    raw["model_label"]
    + " / "
    + raw["backend_label"]
    + " / "
    + raw["serving_mode"]
)

frontier_source = raw[raw["satisfies_sla"]].copy()
frontier_source = frontier_source.sort_values(
    ["cluster_tps_72gpu", "per_gpu_throughput_72gpu"], ascending=[False, False]
)

frontier_indices = []
best_per_gpu = float("-inf")
for idx, row in frontier_source.iterrows():
    per_gpu = float(row["per_gpu_throughput_72gpu"])
    if per_gpu > best_per_gpu:
        frontier_indices.append(idx)
        best_per_gpu = per_gpu

raw["is_global_pareto_frontier"] = raw.index.isin(frontier_indices)
frontier = raw.loc[frontier_indices].sort_values("cluster_tps_72gpu")

raw.to_csv(OUT_CSV, index=False)
frontier.to_csv(OUT_FRONTIER_CSV, index=False)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 7.5))

colors = {
    "Kimi K2.5 NVFP4": "#2563eb",
    "MiniMax M2.7 NVFP4": "#16a34a",
    "GLM-5 NVFP4": "#dc2626",
}
markers = {
    "agg": "o",
    "disagg": "^",
}

for (model_label, mode), group in raw.groupby(["model_label", "serving_mode"]):
    ax.scatter(
        group["cluster_tps_72gpu"],
        group["per_gpu_throughput_72gpu"],
        s=36,
        alpha=0.35,
        color=colors.get(model_label, "#6b7280"),
        marker=markers.get(mode, "s"),
        label=f"{model_label} {mode}",
        edgecolors="none",
    )

if not frontier.empty:
    ax.plot(
        frontier["cluster_tps_72gpu"],
        frontier["per_gpu_throughput_72gpu"],
        color="#111827",
        linewidth=2.2,
        marker="D",
        markersize=6,
        label="Pareto frontier",
        zorder=5,
    )

    for _, row in frontier.iterrows():
        label = f"{row['model_label']}\n{row['backend_label']} {row['serving_mode']}"
        ax.annotate(
            label,
            (row["cluster_tps_72gpu"], row["per_gpu_throughput_72gpu"]),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=8.5,
            color="#111827",
        )

best_rows = (
    raw[raw["satisfies_sla"]]
    .sort_values(["model_label", "per_gpu_throughput_72gpu"], ascending=[True, False])
    .groupby("model_label", as_index=False)
    .head(1)
)
for _, row in best_rows.iterrows():
    ax.scatter(
        [row["cluster_tps_72gpu"]],
        [row["per_gpu_throughput_72gpu"]],
        s=110,
        facecolors="none",
        edgecolors="#111827",
        linewidths=1.8,
        zorder=6,
    )

ax.set_title("GB300 SOTA Model Pareto Frontier", fontsize=16, pad=14)
ax.set_xlabel("TPS (tokens/s)")
ax.set_ylabel("Per-GPU Throughput (tokens/s/GPU)")
ax.ticklabel_format(style="plain", axis="x")
ax.grid(True, color="#e5e7eb")
ax.legend(loc="upper left", fontsize=8.5, frameon=True)
ax.margins(x=0.06, y=0.08)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=180)

print(f"Wrote {OUT_CSV}")
print(f"Wrote {OUT_FRONTIER_CSV}")
print(f"Wrote {OUT_PNG}")
print(f"Rows: {len(raw)}, SLA rows: {int(raw['satisfies_sla'].sum())}, frontier rows: {len(frontier)}")
