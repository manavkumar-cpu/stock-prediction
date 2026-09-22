import json
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from src.logger import get_logger, run_logged

from src.data.download import PROJECT_ROOT, load_config
from src.exceptions import DataValidationError
logger = get_logger("data.validate")
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
REQUIRED_COLUMNS = PRICE_COLUMNS + ["Volume"]


def validate_stock_data(
    data: pd.DataFrame,
    start_date: str,
    end_date_exclusive: str,
    calendar_name: str,
) -> dict:
    report = {
        "rows": len(data),
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    errors = report["errors"]
    warnings = report["warnings"]
    checks = report["checks"]

    if data.empty:
        errors.append("Dataset is empty.")
        return report

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]
    checks["missing_columns"] = missing_columns

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return report

    if not isinstance(data.index, pd.DatetimeIndex):
        errors.append("Index must be a DatetimeIndex.")
        return report

    if data.index.hasnans:
        errors.append("Index contains missing or invalid dates.")
        return report

    if data.index.tz is not None:
        errors.append("Expected timezone-free daily session dates.")
        return report

    if not data.index.equals(data.index.normalize()):
        errors.append("Daily dates contain unexpected time-of-day values.")

    duplicate_count = int(data.index.duplicated().sum())
    checks["duplicate_dates"] = duplicate_count

    if duplicate_count:
        errors.append(f"Found {duplicate_count} duplicate date entries.")

    ordered = bool(data.index.is_monotonic_increasing)
    checks["chronological_order"] = ordered

    if not ordered:
        errors.append("Dates are not in chronological order.")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date_exclusive)

    outside_range = int(
        ((data.index < start) | (data.index >= end)).sum()
    )
    checks["dates_outside_range"] = outside_range

    if outside_range:
        errors.append(f"{outside_range} dates fall outside the requested range.")

    calendar = mcal.get_calendar(calendar_name)
    expected_dates = calendar.valid_days(
        start_date=start,
        end_date=end - pd.Timedelta(days=1),
    ).tz_localize(None)

    observed_dates = data.index.normalize().unique()
    missing_sessions = expected_dates.difference(observed_dates)
    unexpected_sessions = observed_dates.difference(expected_dates)

    checks["expected_sessions"] = len(expected_dates)
    checks["missing_sessions"] = missing_sessions.strftime("%Y-%m-%d").tolist()
    checks["unexpected_sessions"] = (
        unexpected_sessions.strftime("%Y-%m-%d").tolist()
    )

    if len(missing_sessions):
        errors.append(
            f"{len(missing_sessions)} expected trading sessions are absent."
        )

    if len(unexpected_sessions):
        errors.append(
            f"{len(unexpected_sessions)} dates are outside the exchange calendar."
        )

    checks["dtypes"] = {
        column: str(data[column].dtype)
        for column in REQUIRED_COLUMNS
    }

    non_numeric = [
        column for column in REQUIRED_COLUMNS
        if not pd.api.types.is_numeric_dtype(data[column])
        or pd.api.types.is_bool_dtype(data[column])
    ]

    if non_numeric:
        errors.append(f"Non-numeric columns: {non_numeric}")
        return report

    values = data[REQUIRED_COLUMNS]

    checks["missing_values"] = {
        column: int(count)
        for column, count in values.isna().sum().items()
    }

    if values.isna().any().any():
        errors.append("Missing values found in required columns.")

    infinite_count = int(
        np.isinf(values.to_numpy(dtype=float)).sum()
    )
    checks["infinite_values"] = infinite_count

    if infinite_count:
        errors.append(f"Found {infinite_count} infinite values.")

    invalid_prices = int(
        (data[PRICE_COLUMNS] <= 0).any(axis=1).sum()
    )
    checks["rows_with_nonpositive_prices"] = invalid_prices

    if invalid_prices:
        errors.append(f"{invalid_prices} rows contain nonpositive prices.")

    negative_volume = int((data["Volume"] < 0).sum())
    zero_volume = int((data["Volume"] == 0).sum())

    finite_volume = data.loc[np.isfinite(data["Volume"]), "Volume"]
    fractional_volume = int((finite_volume % 1 != 0).sum())

    checks["negative_volume_rows"] = negative_volume
    checks["zero_volume_rows"] = zero_volume
    checks["fractional_volume_rows"] = fractional_volume

    if negative_volume:
        errors.append(f"{negative_volume} rows have negative volume.")

    if fractional_volume:
        errors.append(f"{fractional_volume} rows have fractional volume.")

    if zero_volume:
        warnings.append(f"{zero_volume} rows have zero volume; investigate.")

    tolerance = 1e-8

    inconsistent_ohlc = (
        (data["Low"] > data["High"] + tolerance)
        | (data["Open"] < data["Low"] - tolerance)
        | (data["Open"] > data["High"] + tolerance)
        | (data["Close"] < data["Low"] - tolerance)
        | (data["Close"] > data["High"] + tolerance)
    )

    checks["inconsistent_ohlc_rows"] = int(inconsistent_ohlc.sum())
    checks["inconsistent_ohlc_dates"] = (
        data.index[inconsistent_ohlc].strftime("%Y-%m-%d").tolist()
    )

    if inconsistent_ohlc.any():
        errors.append("OHLC price relationships are inconsistent.")

    return report


def main() -> None:
    config = load_config()
    settings = config["data"]

    ticker = settings["ticker"].strip().upper()
    start = settings["start_date"]
    end = settings["end_date_exclusive"]

    input_path = (
        PROJECT_ROOT
        / settings["raw_dir"]
        / f"{ticker}_{start}_{end}_raw.csv"
    )

    data = pd.read_csv(input_path)

    if "Date" not in data.columns:
        raise ValueError("CSV is missing the Date column.")

    data["Date"] = pd.to_datetime(
        data["Date"], format="%Y-%m-%d", errors="coerce"
    )
    data = data.set_index("Date")

    report = validate_stock_data(
        data=data,
        start_date=start,
        end_date_exclusive=end,
        calendar_name=settings["calendar"],
    )

    if report["errors"]:
        report["status"] = "FAIL"
    elif report["warnings"]:
        report["status"] = "REVIEW"
    else:
        report["status"] = "PASS"

    report["source_file"] = str(input_path.relative_to(PROJECT_ROOT))
    report["calendar"] = settings["calendar"]

    output_dir = PROJECT_ROOT / config["paths"]["metrics"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{ticker}_validation.json"
    report_text = json.dumps(report, indent=2)

    output_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    logger.info("Saved validation report: %s", output_path)

    if report["errors"]:
        raise DataValidationError(
            "Raw-data validation failed: "
            + "; ".join(report["errors"])
        )

    if report["warnings"]:
        logger.warning(
            "Validation requires review: %s",
            "; ".join(report["warnings"]),
        )
    else:
        logger.info("All raw-data validation checks passed")


if __name__ == "__main__":
    run_logged(main, logger)