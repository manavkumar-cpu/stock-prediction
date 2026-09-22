from pathlib import Path
from src.logger import get_logger, run_logged
import pandas as pd
import yaml
import yfinance as yf
from src.exceptions import ConfigurationError, DataDownloadError
logger = get_logger("data.download")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "base.yaml"

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"Configuration file not found: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in: {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigurationError(
            "Configuration must contain a YAML mapping."
        )

    return config

def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    # Add immediately before yf.download(...):
    logger.info(
        "Downloading %s from %s to %s (end exclusive)",
        ticker,
        start_date,
        end_date,                                   
    )
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        multi_level_index=False,
        keepna=True,
        repair=False,
        progress=False,
        threads=False,
        timeout=30,
    )

    if data is None or data.empty:
        raise DataDownloadError(
            f"No data returned for {ticker}. "
            "Check the ticker, date range, internet connection, "
            "and any Yahoo Finance error shown above."
        )

    return data


def main() -> None:
    config = load_config()
    settings = config["data"]

    ticker = settings["ticker"].strip().upper()
    start_date = settings["start_date"]
    end_date = settings["end_date_exclusive"]

    data = download_stock_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=settings["interval"],
        auto_adjust=settings["auto_adjust"],
    )

    print(f"\nTicker: {ticker}")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"First date: {data.index.min()}")
    print(f"Last date: {data.index.max()}")

    print("\nFirst five rows:")
    print(data.head())

    print("\nLast five rows:")
    print(data.tail())

    print("\nDataFrame information:")
    data.info()

    output_dir = PROJECT_ROOT / settings["raw_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"{ticker}_{start_date}_{end_date}_raw.csv"
    )

    data.to_csv(output_path, index=True, index_label="Date")
    logger.info("Saved raw dataset: %s", output_path)

if __name__ == "__main__":
    run_logged(main, logger)