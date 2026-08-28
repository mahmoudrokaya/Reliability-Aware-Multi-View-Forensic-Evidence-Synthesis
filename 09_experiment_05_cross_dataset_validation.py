from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(
    r"D:\47\472\New-Papers\Digital_Forensics_Framework_Network Intrusions"
    r"\Experiments"
)
CLEANING_ROOT = (
    PROJECT_ROOT / "Results" / "Experiment_01_Data_Preparation"
    / "Phase_02_Integrity_Verification" / "Step_02_Data_Cleaning"
)
CLEANING_SUMMARY = CLEANING_ROOT / "Reports" / "Dataset_Cleaning_Summary.csv"
EXP21_ROOT = (
    PROJECT_ROOT / "Results" / "Experiment_03_Framework_Validation"
    / "Experiment_02_Original_Feature_Representation"
    / "Phase_01_Original_Feature_Evaluation"
)
OUT = PROJECT_ROOT / "Results" / "Experiment_05_Cross_Dataset_Validation"
REPORTS = OUT / "Reports"
MANIFESTS = OUT / "Manifests"
LOGS = OUT / "Logs"
for p in (OUT, REPORTS, MANIFESTS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

DATASETS = ("CICIDS2017", "CSE-CIC-IDS2018", "UNSW-NB15", "BoT-IoT")
TARGET = "binary_label"
SEED = 42
DEFAULT_MAX_ROWS = 1_000_000
MIN_RETENTION = 0.50
THRESHOLDS = np.arange(0.05, 0.951, 0.01)

VIEWS = ("volume", "temporal", "transport", "flags_errors", "directional", "general")
TOKENS = {
    "volume": ("byte", "packet", "length", "len", "load", "rate", "total", "mean", "avg", "min", "max", "std"),
    "temporal": ("duration", "dur", "iat", "active", "idle", "time", "jitter"),
    "transport": ("protocol", "proto", "service", "state", "port", "sport", "dport", "window", "header"),
    "flags_errors": ("flag", "fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr", "error", "loss", "retrans"),
    "directional": ("fwd", "forward", "bwd", "backward", "source", "destination", "src", "dst", "inbound", "outbound"),
}
NON_PREDICTIVE = {
    "binary_label", "multiclass_label", "original_label",
    "source_dataset", "source_file", "source_row",
}
IDENTIFIERS = (
    r"(^|_)flow_id($|_)", r"(^|_)src_ip($|_)", r"(^|_)dst_ip($|_)",
    r"(^|_)source_ip($|_)", r"(^|_)destination_ip($|_)", r"(^|_)record_id($|_)",
)
TIMESTAMPS = (r"(^|_)timestamp($|_)", r"(^|_)time_stamp($|_)")

DESCRIPTORS = (
    "present", "observed_fraction", "numeric_fraction",
    "zero_fraction", "positive_fraction", "negative_fraction",
    "log_abs_mean", "log_abs_std", "log_abs_median", "log_abs_max",
    "categorical_observed_fraction",
)
DESCRIPTOR_COLUMNS = tuple(f"{v}__{d}" for v in VIEWS for d in DESCRIPTORS)

logger = logging.getLogger("experiment5")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
fh = logging.FileHandler(LOGS / "experiment_05_cross_dataset.log", mode="w", encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)

@dataclass
class TransferResult:
    source_dataset: str
    target_dataset: str
    source_train_rows: int
    source_validation_rows: int
    source_test_rows: int
    target_test_rows: int
    descriptor_count: int
    threshold: float
    source_test_balanced_accuracy: float
    source_test_f1: float
    source_test_mcc: float
    source_test_roc_auc: float
    target_accuracy: float
    target_balanced_accuracy: float
    target_precision: float
    target_recall: float
    target_f1: float
    target_mcc: float
    target_roc_auc: float
    target_average_precision: float
    delta_balanced_accuracy: float
    delta_f1: float
    delta_mcc: float
    delta_roc_auc: float
    training_seconds: float
    target_inference_seconds: float


def norm(x):
    s = re.sub(r"[^a-z0-9_]+", "_", str(x).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def matches(name, patterns):
    n = norm(name)
    return any(re.search(p, n) for p in patterns)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def discover_parts(dataset):
    d = CLEANING_ROOT / dataset / "Cleaned_Data"
    pq = sorted(d.glob("cleaned_part_*.parquet"))
    return pq if pq else sorted(d.glob("cleaned_part_*.csv.gz"))


def read_part(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, compression="gzip", low_memory=False)


def count_rows(path):
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    n = 0
    for c in pd.read_csv(path, compression="gzip", usecols=[0], chunksize=250_000):
        n += len(c)
    return n


def allocate(sizes, requested):
    total = sum(sizes)
    if requested <= 0 or requested >= total:
        return sizes.copy()
    exact = [requested * s / total for s in sizes]
    a = [min(s, int(np.floor(x))) for s, x in zip(sizes, exact)]
    rem = requested - sum(a)
    for i in np.argsort([x - np.floor(x) for x in exact])[::-1]:
        if rem <= 0:
            break
        if a[i] < sizes[i]:
            a[i] += 1
            rem -= 1
    return a


def load_sample(dataset, max_rows, seed):
    parts = discover_parts(dataset)
    if not parts:
        raise FileNotFoundError(f"No cleaned parts found for {dataset}")
    sizes = [count_rows(p) for p in parts]
    total = sum(sizes)
    requested = total if max_rows <= 0 else min(max_rows, total)
    alloc = allocate(sizes, requested)
    frames = []
    for i, (p, size, n) in enumerate(zip(parts, sizes, alloc), 1):
        if n <= 0:
            continue
        f = read_part(p)
        if n < size:
            f = f.sample(n=n, random_state=seed + i)
        frames.append(f)
        logger.info("%s | loaded part %d/%d | %d/%d rows", dataset, i, len(parts), len(f), size)
    df = pd.concat(frames, ignore_index=True, sort=False)
    if len(df) > requested:
        df = df.sample(n=requested, random_state=seed).reset_index(drop=True)
    return df, total


def select_predictors(df):
    if TARGET not in df:
        raise ValueError(f"Missing target: {TARGET}")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    ok = y.isin([0, 1])
    df = df.loc[ok].copy()
    y = y.loc[ok].astype("int8")
    excluded = set(NON_PREDICTIVE)
    for c in df.columns:
        if matches(c, IDENTIFIERS) or matches(c, TIMESTAMPS):
            excluded.add(c)
    X = df[[c for c in df.columns if c not in excluded]].copy()
    removable = [c for c in X if X[c].isna().all() or X[c].nunique(dropna=True) <= 1]
    if removable:
        X = X.drop(columns=removable)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_splits(dataset, expected_rows):
    p = EXP21_ROOT / dataset / "Splits" / "fixed_split_assignments.csv"
    a = pd.read_csv(p).sort_values("row_position")
    if len(a) != expected_rows:
        raise ValueError(
            f"{dataset}: sample/split mismatch: sample={expected_rows}, split={len(a)}. "
            "Use the same --max-rows value used in Experiment 2.1."
        )
    return {
        n: a.loc[a["split"].eq(n), "row_position"].to_numpy(dtype=int)
        for n in ("train", "validation", "test")
    }


def build_views(X):
    assigned = set()
    views = {}
    for name, tokens in TOKENS.items():
        cols = sorted({c for c in X.columns if any(token in norm(c) for token in tokens)})
        if cols:
            views[name] = cols
            assigned.update(cols)
    residual = [c for c in X.columns if c not in assigned]
    if residual:
        views["general"] = residual
    return views


def numeric_descriptors(frame):
    rows = len(frame)
    keys = (
        "numeric_fraction", "zero_fraction", "positive_fraction",
        "negative_fraction", "log_abs_mean", "log_abs_std",
        "log_abs_median", "log_abs_max",
    )
    if frame.shape[1] == 0:
        z = np.zeros(rows, dtype=np.float32)
        return {k: z.copy() for k in keys}
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    v = numeric.to_numpy(dtype=np.float64, copy=False)
    obs = np.isfinite(v)
    count = obs.sum(axis=1)
    den = np.maximum(count, 1)
    safe = np.where(obs, v, np.nan)
    zero = (obs & np.isclose(v, 0.0, atol=1e-12)).sum(axis=1) / den
    pos = (obs & (v > 0)).sum(axis=1) / den
    neg = (obs & (v < 0)).sum(axis=1) / den
    log_abs = np.log1p(np.abs(safe))
    with np.errstate(all="ignore"):
        mean = np.nanmean(log_abs, axis=1)
        std = np.nanstd(log_abs, axis=1)
        med = np.nanmedian(log_abs, axis=1)
        mx = np.nanmax(log_abs, axis=1)
    clean = lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return {
        "numeric_fraction": (count / max(frame.shape[1], 1)).astype(np.float32),
        "zero_fraction": zero.astype(np.float32),
        "positive_fraction": pos.astype(np.float32),
        "negative_fraction": neg.astype(np.float32),
        "log_abs_mean": clean(mean),
        "log_abs_std": clean(std),
        "log_abs_median": clean(med),
        "log_abs_max": clean(mx),
    }


def semantic_descriptors(X, views):
    rows = len(X)
    out = {}
    for view in VIEWS:
        cols = views.get(view, [])
        if not cols:
            z = np.zeros(rows, dtype=np.float32)
            for d in DESCRIPTORS:
                out[f"{view}__{d}"] = z.copy()
            continue
        vf = X[cols]
        out[f"{view}__present"] = np.ones(rows, dtype=np.float32)
        out[f"{view}__observed_fraction"] = 1.0 - vf.isna().mean(axis=1).to_numpy(dtype=np.float32)
        numeric_cols = list(vf.select_dtypes(include=[np.number, "bool"]).columns)
        categorical_cols = [c for c in cols if c not in numeric_cols]
        nd = numeric_descriptors(vf[numeric_cols] if numeric_cols else pd.DataFrame(index=vf.index))
        for k, arr in nd.items():
            out[f"{view}__{k}"] = arr
        if categorical_cols:
            cat_obs = 1.0 - vf[categorical_cols].isna().mean(axis=1).to_numpy(dtype=np.float32)
        else:
            cat_obs = np.zeros(rows, dtype=np.float32)
        out[f"{view}__categorical_observed_fraction"] = cat_obs
    return pd.DataFrame(out).reindex(columns=DESCRIPTOR_COLUMNS, fill_value=0.0).astype(np.float32)


def metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(np.int8)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def select_threshold(y, p):
    best = None
    best_t = 0.5
    best_m = None
    for t in THRESHOLDS:
        m = metrics(y, p, float(t))
        key = (m["balanced_accuracy"], m["f1"], m["mcc"], m["roc_auc"], -abs(float(t) - 0.5))
        if best is None or key > best:
            best = key
            best_t = float(t)
            best_m = m
    return best_t, best_m


def create_model(seed):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def readiness():
    summary = pd.read_csv(CLEANING_SUMMARY)
    records = []
    ready = []
    for d in DATASETS:
        m = summary[summary["dataset"].astype(str).eq(d)]
        reason = ""
        ok = False
        if m.empty:
            reason = "No cleaning summary"
        else:
            r = m.iloc[-1]
            status = str(r.get("status", "")).upper()
            inp = int(r.get("input_rows", 0))
            out = int(r.get("output_rows", 0))
            ratio = out / inp if inp > 0 else 0.0
            split = EXP21_ROOT / d / "Splits" / "fixed_split_assignments.csv"
            if status != "PASS":
                reason = f"Cleaning status={status}"
            elif ratio < MIN_RETENTION:
                reason = f"Retention ratio={ratio:.4f}"
            elif not discover_parts(d):
                reason = "No cleaned parts"
            elif not split.exists():
                reason = "No Experiment 2.1 split assignments"
            else:
                ok = True
                reason = f"Ready; retention ratio={ratio:.4f}"
        records.append({"dataset": d, "ready": ok, "reason": reason})
        if ok:
            ready.append(d)
    pd.DataFrame(records).to_csv(REPORTS / "Dataset_Readiness.csv", index=False)
    return ready


def prepare(dataset, max_rows, seed):
    logger.info("Preparing %s", dataset)
    raw, total = load_sample(dataset, max_rows, seed)
    X, y = select_predictors(raw)
    splits = load_splits(dataset, len(X))
    views = build_views(X)
    desc = semantic_descriptors(X, views)
    logger.info(
        "%s | rows=%d | predictors=%d | views=%s | descriptors=%d",
        dataset, len(X), X.shape[1], ",".join(views.keys()), desc.shape[1]
    )
    return {"name": dataset, "X": X, "y": y, "splits": splits, "views": views, "desc": desc, "total": total}


def run_pair(source, target, seed):
    sname, tname = source["name"], target["name"]
    pair = f"{sname}_to_{tname}"
    root = OUT / pair
    for sub in ("Predictions", "Metrics", "Models", "Manifests"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    si, ti = source["splits"], target["splits"]
    Xtr, ytr = source["desc"].iloc[si["train"]], source["y"].iloc[si["train"]]
    Xva, yva = source["desc"].iloc[si["validation"]], source["y"].iloc[si["validation"]]
    Xst, yst = source["desc"].iloc[si["test"]], source["y"].iloc[si["test"]]
    Xtt, ytt = target["desc"].iloc[ti["test"]], target["y"].iloc[ti["test"]]

    model = create_model(seed)
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_seconds = time.perf_counter() - t0

    pva = model.predict_proba(Xva)[:, 1]
    threshold, _ = select_threshold(yva, pva)
    pst = model.predict_proba(Xst)[:, 1]
    sm = metrics(yst, pst, threshold)

    t0 = time.perf_counter()
    ptt = model.predict_proba(Xtt)[:, 1]
    inference_seconds = time.perf_counter() - t0
    tm = metrics(ytt, ptt, threshold)

    logger.info(
        "%s | threshold=%.3f | target BA=%.6f | F1=%.6f | MCC=%.6f | AUC=%.6f",
        pair, threshold, tm["balanced_accuracy"], tm["f1"], tm["mcc"], tm["roc_auc"]
    )

    pd.DataFrame({
        "true_label": ytt.to_numpy(dtype=np.int8),
        "attack_probability": ptt,
        "predicted_label": (ptt >= threshold).astype(np.int8),
    }).to_csv(root / "Predictions" / "target_test_predictions.csv", index=False)

    with (root / "Models" / "source_transfer_model.pkl").open("wb") as f:
        pickle.dump(model, f)

    save_json(
        root / "Manifests" / "transfer_manifest.json",
        {
            "source_dataset": sname,
            "target_dataset": tname,
            "descriptor_schema": list(DESCRIPTOR_COLUMNS),
            "source_views": source["views"],
            "target_views": target["views"],
            "raw_feature_equivalence_assumed": False,
            "target_model_refit": False,
            "target_statistics_fitted": False,
            "target_labels_used_for_adaptation": False,
            "threshold_selection": "source validation only",
            "threshold": threshold,
        },
    )

    result = TransferResult(
        source_dataset=sname,
        target_dataset=tname,
        source_train_rows=len(si["train"]),
        source_validation_rows=len(si["validation"]),
        source_test_rows=len(si["test"]),
        target_test_rows=len(ti["test"]),
        descriptor_count=len(DESCRIPTOR_COLUMNS),
        threshold=threshold,
        source_test_balanced_accuracy=sm["balanced_accuracy"],
        source_test_f1=sm["f1"],
        source_test_mcc=sm["mcc"],
        source_test_roc_auc=sm["roc_auc"],
        target_accuracy=tm["accuracy"],
        target_balanced_accuracy=tm["balanced_accuracy"],
        target_precision=tm["precision"],
        target_recall=tm["recall"],
        target_f1=tm["f1"],
        target_mcc=tm["mcc"],
        target_roc_auc=tm["roc_auc"],
        target_average_precision=tm["average_precision"],
        delta_balanced_accuracy=tm["balanced_accuracy"] - sm["balanced_accuracy"],
        delta_f1=tm["f1"] - sm["f1"],
        delta_mcc=tm["mcc"] - sm["mcc"],
        delta_roc_auc=tm["roc_auc"] - sm["roc_auc"],
        training_seconds=train_seconds,
        target_inference_seconds=inference_seconds,
    )

    pd.DataFrame([asdict(result)]).to_csv(root / "Metrics" / "transfer_result.csv", index=False)
    save_json(root / "Metrics" / "transfer_result.json", asdict(result))
    return result


def save_reports(results):
    frame = pd.DataFrame([asdict(r) for r in results])
    frame.to_csv(REPORTS / "Cross_Dataset_Results.csv", index=False)
    if frame.empty:
        return
    for metric, name in (
        ("target_balanced_accuracy", "Cross_Dataset_Matrix_Balanced_Accuracy.csv"),
        ("target_f1", "Cross_Dataset_Matrix_F1.csv"),
        ("target_mcc", "Cross_Dataset_Matrix_MCC.csv"),
        ("target_roc_auc", "Cross_Dataset_Matrix_ROC_AUC.csv"),
    ):
        frame.pivot(index="source_dataset", columns="target_dataset", values=metric).to_csv(REPORTS / name)
    cols = [
        "source_dataset", "target_dataset",
        "source_test_balanced_accuracy", "target_balanced_accuracy", "delta_balanced_accuracy",
        "source_test_f1", "target_f1", "delta_f1",
        "source_test_mcc", "target_mcc", "delta_mcc",
        "source_test_roc_auc", "target_roc_auc", "delta_roc_auc",
    ]
    frame[cols].to_csv(REPORTS / "Within_vs_Cross_Dataset.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=DATASETS, default=None)
    parser.add_argument("--target", choices=DATASETS, default=None)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if (args.source is None) != (args.target is None):
        raise ValueError("--source and --target must be supplied together")
    if args.source == args.target and args.source is not None:
        raise ValueError("Source and target must differ")

    logger.info("=" * 78)
    logger.info("EXPERIMENT 5: CROSS-DATASET VALIDATION")
    logger.info("Fixed semantic descriptor schema: %d columns", len(DESCRIPTOR_COLUMNS))
    logger.info("Target adaptation/refitting: NONE")
    logger.info("=" * 78)

    ready = readiness()
    if args.source is not None:
        if args.source not in ready or args.target not in ready:
            logger.error("Requested pair is not fully runnable. Ready=%s", ready)
            return 1
        pairs = [(args.source, args.target)]
    else:
        if len(ready) < 2:
            logger.error("Need at least two runnable datasets. Ready=%s", ready)
            return 1
        pairs = [(s, t) for s in ready for t in ready if s != t]

    required = sorted({d for pair in pairs for d in pair})
    cache = {d: prepare(d, args.max_rows, args.seed) for d in required}

    results = []
    failures = []
    for s, t in pairs:
        try:
            results.append(run_pair(cache[s], cache[t], args.seed))
        except Exception:
            failures.append((s, t))
            logger.exception("Transfer failed: %s -> %s", s, t)

    save_reports(results)
    save_json(
        MANIFESTS / "experiment_05_run_manifest.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "runnable_datasets": ready,
            "pairs": [{"source": s, "target": t} for s, t in pairs],
            "successful_pairs": [{"source": r.source_dataset, "target": r.target_dataset} for r in results],
            "failed_pairs": [{"source": s, "target": t} for s, t in failures],
            "max_rows": args.max_rows,
            "seed": args.seed,
            "semantic_views": list(VIEWS),
            "descriptors_per_view": list(DESCRIPTORS),
            "descriptor_count": len(DESCRIPTOR_COLUMNS),
            "raw_feature_equivalence_assumed": False,
            "target_statistics_fitted": False,
            "target_labels_used_for_adaptation": False,
            "target_model_refit": False,
            "threshold_selection": "source validation only",
        },
    )

    logger.info("Successful transfers=%d | Failed=%d", len(results), len(failures))
    logger.info("Results: %s", OUT)
    return 0 if results and not failures else (2 if results else 1)


if __name__ == "__main__":
    raise SystemExit(main())
