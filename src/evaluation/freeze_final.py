import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.download import PROJECT_ROOT
from src.exceptions import DataValidationError
from src.logger import get_logger, run_logged


logger = get_logger("evaluation.freeze_final")

PREPARED = (
    "data/processed/final_development/"
    "20260920T112216_338230Z"
)
NEURAL = "artifacts/final_neural/20260920T112518_629227Z"
XGBOOST = "artifacts/final_xgboost/20260920T115749_481182Z"
ARIMA = "artifacts/final_arima/20260920T120716_481620Z"

EXPECTED_NEURAL = {
    "LSTM_h1_w120": ("h1_w120", 1, 120, 35),
    "Attention_LSTM_h1_w120": ("h1_w120", 1, 120, 10),
    "GRU_h1_w60": ("h1_w60", 1, 60, 9),
    "LSTM_h5_w120": ("h5_w120", 5, 120, 34),
}


def require(condition, message):
    if not condition:
        raise DataValidationError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    prepared = PROJECT_ROOT / PREPARED
    neural = PROJECT_ROOT / NEURAL
    xgboost = PROJECT_ROOT / XGBOOST
    arima = PROJECT_ROOT / ARIMA

    config = read_json(prepared / "base_config_snapshot.json")
    settings = config["data"]

    require(
        config["split"] == {
            "train_end": "2019-12-31",
            "validation_end": "2022-12-31",
            "test_end": "2025-12-31",
        },
        "Unexpected split configuration.",
    )

    tracked = {}

    def track(path):
        path = Path(path).resolve()
        require(path.is_file(), f"Required file is missing: {path}")
        relative = str(path.relative_to(PROJECT_ROOT.resolve()))
        tracked[relative] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        return relative

    def same_file(first, second):
        require(
            sha256(first) == sha256(second),
            f"Snapshot mismatch: {first} versus {second}",
        )

    track(prepared / "base_config_snapshot.json")
    track(prepared / "summary.csv")

    for dataset in ("h1_w120", "h1_w60", "h5_w120"):
        folder = prepared / dataset
        for filename in (
            "experiment.json",
            "preprocessing.json",
            "feature_scaler.pkl",
            "train.npz",
            "validation.npz",
            "train_metadata.csv",
            "validation_metadata.csv",
        ):
            track(folder / filename)

    selected_neural = read_json(neural / "selected_models.json")
    require(
        len(selected_neural) == 4
        and {entry["model"] for entry in selected_neural}
        == set(EXPECTED_NEURAL),
        "Unexpected neural model registry.",
    )

    models = []

    for entry in selected_neural:
        name = entry["model"]
        dataset, horizon, window, epoch = EXPECTED_NEURAL[name]

        require(
            entry["dataset"] == dataset
            and entry["horizon"] == horizon
            and entry["window"] == window
            and entry["best_epoch"] == epoch,
            f"{name}: selected settings differ from reviewed results.",
        )

        checkpoint = neural / entry["checkpoint"]
        require(
            sha256(checkpoint) == entry["checkpoint_sha256"],
            f"{name}: checkpoint fingerprint mismatch.",
        )

        folder = checkpoint.parent
        for filename in (
            "experiment.json", "preprocessing.json", "feature_scaler.pkl"
        ):
            track(folder / filename)
            same_file(folder / filename, prepared / dataset / filename)

        track(folder / "history.csv")

        models.append({
            **entry,
            "checkpoint": track(checkpoint),
            "preprocessing": str(
                (folder / "preprocessing.json").relative_to(PROJECT_ROOT)
            ),
            "feature_scaler": str(
                (folder / "feature_scaler.pkl").relative_to(PROJECT_ROOT)
            ),
            "output": "scaled_dollar_correction",
        })

    for filename in (
        "selected_models.json", "run_info.json",
        "training_summary.csv", "validation_metrics.csv",
    ):
        track(neural / filename)

    selected_xgb = read_json(xgboost / "selected_model.json")
    require(
        selected_xgb["depth"] == 4
        and selected_xgb["best_rounds"] == 98
        and selected_xgb["prediction_iteration_range"] == [0, 98]
        and selected_xgb["horizon"] == 1
        and selected_xgb["window"] == 120,
        "XGBoost selection differs from reviewed results.",
    )

    xgb_checkpoint = xgboost / selected_xgb["checkpoint"]
    require(
        sha256(xgb_checkpoint) == selected_xgb["checkpoint_sha256"],
        "XGBoost checkpoint fingerprint mismatch.",
    )

    for filename in (
        "experiment.json", "preprocessing.json", "feature_scaler.pkl"
    ):
        track(xgboost / filename)
        same_file(xgboost / filename, prepared / "h1_w120" / filename)

    for filename in (
        "selected_model.json", "run_info.json",
        "candidate_metrics.csv", "validation_metrics.csv",
    ):
        track(xgboost / filename)

    models.append({
        **selected_xgb,
        "dataset": "h1_w120",
        "checkpoint": track(xgb_checkpoint),
        "preprocessing": f"{XGBOOST}/preprocessing.json",
        "feature_scaler": f"{XGBOOST}/feature_scaler.pkl",
        "output": "dollar_correction",
    })

    selected_arima = read_json(arima / "selected_model.json")
    require(
        selected_arima["order"] == [1, 1, 1]
        and selected_arima["trend"] == "n"
        and selected_arima["coefficient_fit_end"] == "2019-12-31"
        and selected_arima["state_as_of"] == "2022-12-30",
        "ARIMA selection differs from reviewed results.",
    )

    arima_checkpoint = arima / selected_arima["checkpoint"]
    require(
        sha256(arima_checkpoint) == selected_arima["checkpoint_sha256"],
        "ARIMA checkpoint fingerprint mismatch.",
    )

    for filename in (
        "selected_model.json", "run_info.json", "experiment.json",
        "candidate_metrics.csv", "validation_metrics.csv",
    ):
        track(arima / filename)

    same_file(
        arima / "experiment.json",
        prepared / "h1_w120" / "experiment.json",
    )

    models.append({
        **selected_arima,
        "checkpoint": track(arima_checkpoint),
        "output": "price_usd",
    })

    ticker = settings["ticker"].strip().upper()
    stem = (
        f"{ticker}_{settings['start_date']}_"
        f"{settings['end_date_exclusive']}"
    )

    raw_path = track(
        PROJECT_ROOT / settings["raw_dir"] / f"{stem}_raw.csv"
    )
    feature_path = track(
        PROJECT_ROOT / settings["processed_dir"] / f"{stem}_features.csv"
    )

    # Record the implementation and environment used to select the models.
    for filename in (
        "requirements-lock.txt",
        "docs/final_evaluation_protocol.md",
        "src/features/build.py",
        "src/data/prepare_final_development.py",
        "src/models/train_final_neural.py",
        "src/models/train_final_xgboost.py",
        "src/evaluation/final_arima.py",
        "src/evaluation/metrics.py",
        "src/evaluation/freeze_final.py",
    ):
        track(PROJECT_ROOT / filename)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        PROJECT_ROOT / config["paths"]["artifacts"]
        / "final_evaluation" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_utc": stamp,
        "status": "models_frozen_test_not_evaluated",
        "ticker": ticker,
        "prepared_data": PREPARED,
        "raw_data": raw_path,
        "feature_data": feature_path,
        "split": config["split"],
        "models": models,
        "baselines": {
            "naive": "forecast-date Close for each horizon",
            "training_mean_change": {
                "h1": f"{PREPARED}/h1_w120/preprocessing.json",
                "h5": f"{PREPARED}/h5_w120/preprocessing.json",
            },
        },
        "test_policy": {
            "refit_models": False,
            "refit_preprocessing": False,
            "arima_update_state_after_observation": True,
            "match_dates_within_horizon": True,
            "h5_target_span_must_remain_in_test_partition": True,
            "test_partition": "2023–2025 as one continuous period",
        },
        "files": tracked,
    }

    manifest_path = output_dir / "frozen_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    manifest_hash = sha256(manifest_path)
    (output_dir / "frozen_manifest.sha256").write_text(
        manifest_hash + "\n", encoding="utf-8"
    )

    print("\nFrozen models:")
    for model in models:
        print(f"- {model['model']}")

    print(f"\nFingerprinted files: {len(tracked)}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print(f"Saved manifest: {manifest_path}")

    logger.info("Checkpoint and preprocessing snapshot checks PASS")
    logger.info("No test predictions or metrics calculated")


if __name__ == "__main__":
    run_logged(main, logger)