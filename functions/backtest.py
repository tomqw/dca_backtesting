import csv
import polars as pl
import numpy as np
from datetime import datetime, timezone
from functions.fetch_data import folder
from functions.config import get_trading_fee_percent, get_results_dir
import rust_backend


def readPriceData(pair):
    return pl.read_parquet(folder + pair + ".parquet")


def saveResult(results, file_name):
    data_file = open(get_results_dir() + file_name, "w")
    csv_writer = csv.writer(data_file)

    count = 0

    for result in results:
        if count == 0:
            header = result.keys()
            csv_writer.writerow(header)
            count += 1

        csv_writer.writerow(result.values())

    data_file.close()


def startBacktest(
    config, pair, startDate="2024-01-01 00:00:00", endDate="2024-01-31 00:00:00", trading_fee=None
):
    if startDate == "":
        raise Exception("Start Date not defined")
    if endDate == "":
        raise Exception("End Date not defined")

    if trading_fee is None:
        trading_fee = get_trading_fee_percent()

    df = pl.read_parquet(folder + pair + ".parquet")

    epoch = df["epoch"].to_numpy()
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    close = df["close"].to_numpy().astype(np.float64)

    start_dt = datetime.fromisoformat(startDate).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(endDate).replace(tzinfo=timezone.utc)
    start_epoch = int(start_dt.timestamp() * 1_000_000)
    end_epoch = int(end_dt.timestamp() * 1_000_000)

    result = rust_backend.run_backtest(
        epoch, high, low, close,
        start_epoch, end_epoch,
        float(config.max_amount_for_bot_usage),
        float(trading_fee),
        float(config.take_profit),
        float(config.base_order),
        float(config.safety_order),
        int(config.max_safety_orders),
        float(config.deviation_to_open_safety_order),
        float(config.safety_order_volume_scale),
        float(config.safety_order_step_scale),
    )

    result["config_name"] = config.config_name
    result["pair"] = pair
    result["max_amount_for_bot_usage"] = config.max_amount_for_bot_usage
    result["max_safety_order_price_deviation"] = config.max_safety_order_price_deviation
    result["base_order"] = config.base_order
    result["safety_order"] = config.safety_order
    result["deviation_to_open_safety_order"] = config.deviation_to_open_safety_order
    result["safety_order_volume_scale"] = config.safety_order_volume_scale
    result["safety_order_step_scale"] = config.safety_order_step_scale

    return result
