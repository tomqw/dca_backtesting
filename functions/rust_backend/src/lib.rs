mod backtest;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn run_backtest(
    py: Python<'_>,
    epoch: PyReadonlyArray1<i64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    start_epoch: i64,
    end_epoch: i64,
    start_capital: f64,
    trading_fee: f64,
    take_profit: f64,
    base_order: f64,
    safety_order: f64,
    max_safety_orders: i64,
    deviation_to_open_safety_order: f64,
    safety_order_volume_scale: f64,
    safety_order_step_scale: f64,
) -> PyResult<PyObject> {
    let epoch_slice = epoch.as_slice()?;
    let high_slice = high.as_slice()?;
    let low_slice = low.as_slice()?;
    let close_slice = close.as_slice()?;

    let cfg = backtest::Config {
        start_capital,
        trading_fee,
        take_profit,
        base_order,
        safety_order,
        max_safety_orders,
        deviation_to_open_safety_order,
        safety_order_volume_scale,
        safety_order_step_scale,
        start_epoch,
        end_epoch,
    };

    let results = backtest::run_backtest_loop(
        epoch_slice,
        high_slice,
        low_slice,
        close_slice,
        &cfg,
    );

    let avg_so = if results.total_deals > 0 {
        ((results.total_so as f64 / results.total_deals as f64) * 100.0).round() / 100.0
    } else {
        0.0
    };

    let avaiable_capital = (results.avaiable_capital * 100.0).round() / 100.0;
    let bot_total_volume = (results.bot_total_volume * 100.0).round() / 100.0;
    let bot_capital = (results.bot_total_coins * results.last_close * 100.0).round() / 100.0;
    let bot_current_profit = ((bot_capital - bot_total_volume) * 100.0).round() / 100.0;

    let bot_current_profit_percent = if bot_current_profit != 0.0 && bot_total_volume != 0.0 {
        (((100.0 / bot_total_volume * bot_capital) * 100.0).round() / 100.0) - 100.0
    } else {
        0.0
    };

    let mut avg_deal_time: f64 = 0.0;
    let max_deal_time: f64;
    if !results.deal_times.is_empty() {
        let sum: f64 = results.deal_times.iter().sum();
        avg_deal_time = ((sum / results.deal_times.len() as f64) * 100.0).round() / 100.0;
        max_deal_time = results.max_deal_time;
    } else {
        avg_deal_time = 0.0;
        max_deal_time = 0.0;
    }

    let total_capital = ((avaiable_capital + bot_capital) * 100.0).round() / 100.0;
    let profit = ((total_capital - start_capital) * 100.0).round() / 100.0;
    let profit_percent = if start_capital != 0.0 {
        (((100.0 / start_capital * profit) * 100.0).round() / 100.0)
    } else {
        0.0
    };

    let backtest_start_str = if results.backtest_start_epoch != 0 {
        let naive = chrono::NaiveDateTime::from_timestamp_micros(results.backtest_start_epoch)
            .unwrap_or_default();
        naive.format("%Y-%m-%d %H:%M:%S").to_string()
    } else {
        String::new()
    };

    let backtest_end_str = if results.backtest_end_epoch != 0 {
        let naive = chrono::NaiveDateTime::from_timestamp_micros(results.backtest_end_epoch)
            .unwrap_or_default();
        naive.format("%Y-%m-%d %H:%M:%S").to_string()
    } else {
        String::new()
    };

    let dict = PyDict::new_bound(py);
    dict.set_item("total_capital", total_capital)?;
    dict.set_item("profit", profit)?;
    dict.set_item("profit_percent", profit_percent)?;
    dict.set_item("bot_total_volume", bot_total_volume)?;
    dict.set_item("bot_current_profit", bot_current_profit)?;
    dict.set_item("bot_current_profit_percent", bot_current_profit_percent)?;
    dict.set_item("bot_avg_price", (results.bot_avg_price * 100.0).round() / 100.0)?;
    dict.set_item("bot_sell_price", (results.sell_price * 100.0).round() / 100.0)?;
    dict.set_item("avaiable_capital", avaiable_capital)?;
    dict.set_item("total_deals", results.total_deals)?;
    dict.set_item("highest_so", results.highest_so)?;
    dict.set_item("avg_so", avg_so)?;
    dict.set_item("backtest_start", backtest_start_str)?;
    dict.set_item("backtest_end", backtest_end_str)?;
    dict.set_item("max_deal_time", max_deal_time)?;
    dict.set_item("avg_deal_time", avg_deal_time)?;

    Ok(dict.into())
}

#[pymodule]
fn rust_backend(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    Ok(())
}
