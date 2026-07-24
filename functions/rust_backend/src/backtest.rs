#[derive(Clone)]
pub struct Config {
    pub start_capital: f64,
    pub trading_fee: f64,
    pub take_profit: f64,
    pub base_order: f64,
    pub safety_order: f64,
    pub max_safety_orders: i64,
    pub deviation_to_open_safety_order: f64,
    pub safety_order_volume_scale: f64,
    pub safety_order_step_scale: f64,
    pub start_epoch: i64,
    pub end_epoch: i64,
}

pub struct Results {
    pub avaiable_capital: f64,
    pub bot_total_volume: f64,
    pub bot_total_coins: f64,
    pub bot_avg_price: f64,
    pub sell_price: f64,
    pub total_deals: i64,
    pub highest_so: i64,
    pub total_so: i64,
    pub max_deal_time: f64,
    pub backtest_start_epoch: i64,
    pub backtest_end_epoch: i64,
    pub last_close: f64,
    pub deal_times: Vec<f64>,
}

pub fn run_backtest_loop(
    epoch: &[i64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    cfg: &Config,
) -> Results {
    let mut avaiable_capital = cfg.start_capital;
    let mut start = false;
    let mut stop = false;
    let mut total_deals: i64 = 0;
    let mut highest_so: i64 = 0;
    let mut total_so: i64 = 0;
    let mut max_deal_time: f64 = 0.0;
    let mut backtest_start_epoch: i64 = 0;
    let mut backtest_end_epoch: i64 = 0;
    let mut deal_times: Vec<f64> = Vec::new();

    let mut reset_data = true;

    let mut bot_total_volume: f64 = 0.0;
    let mut bot_total_coins: f64 = 0.0;
    let mut bot_avg_price: f64 = 0.0;
    let mut next_so_buy_price: f64 = 0.0;
    let mut sell_price: f64 = 0.0;
    let mut deal_start_price: f64 = 0.0;
    let mut safety_order_amount: f64 = 0.0;
    let mut safety_order_deviation: f64 = 0.0;
    let mut current_safety_order: i64 = 0;
    let mut total_deviation: f64 = 0.0;
    let mut deal_start_epoch: i64 = 0;
    let mut last_processed_close: f64 = 0.0;
    let mut last_processed_epoch: i64 = 0;

    let n = epoch.len();

    for i in 0..n {
        let price_epoch = epoch[i];

        if !start {
            if price_epoch >= cfg.start_epoch {
                start = true;
                backtest_start_epoch = price_epoch;
            }
        }

        if cfg.end_epoch != 0 {
            if !stop {
                if price_epoch >= cfg.end_epoch {
                    stop = true;
                    backtest_end_epoch = price_epoch;
                }
            }
        }

        if start && !stop {
            let high_price = high[i];
            let low_price = low[i];
            let close_price = close[i];
            last_processed_close = close_price;
            last_processed_epoch = price_epoch;

            if reset_data {
                bot_total_volume = 0.0;
                bot_total_coins = 0.0;
                bot_avg_price = 0.0;
                next_so_buy_price = 0.0;
                sell_price = 0.0;
                deal_start_price = 0.0;
                safety_order_amount = 0.0;
                safety_order_deviation = 0.0;
                current_safety_order = 0;
                total_deviation = 0.0;
                reset_data = false;
                deal_start_epoch = price_epoch;
            }

            if bot_total_volume == 0.0 {
                total_deals += 1;
                let buy_amount = cfg.base_order / close_price;
                bot_avg_price = close_price;
                deal_start_price = close_price;
                bot_total_coins += buy_amount;
                next_so_buy_price = deal_start_price
                    - (deal_start_price / 100.0 * cfg.deviation_to_open_safety_order);

                total_deviation = cfg.deviation_to_open_safety_order;
                safety_order_deviation =
                    cfg.deviation_to_open_safety_order * cfg.safety_order_step_scale;

                sell_price = close_price + (close_price / 100.0 * cfg.take_profit);
                bot_total_volume = cfg.base_order;

                avaiable_capital -= cfg.base_order;
                avaiable_capital -= cfg.base_order / 100.0 * cfg.trading_fee;
            } else {
                if low_price <= next_so_buy_price {
                    if current_safety_order < cfg.max_safety_orders {
                        let buy_price = next_so_buy_price;
                        total_deviation += safety_order_deviation;

                        if current_safety_order == 0 {
                            safety_order_amount = cfg.safety_order;
                        } else {
                            safety_order_amount *= cfg.safety_order_volume_scale;
                        }

                        avaiable_capital -= safety_order_amount;
                        avaiable_capital -= safety_order_amount / 100.0 * cfg.trading_fee;

                        let buy_amount = safety_order_amount / buy_price;
                        bot_total_volume += safety_order_amount;
                        current_safety_order += 1;
                        bot_total_coins += buy_amount;

                        if current_safety_order < cfg.max_safety_orders {
                            next_so_buy_price = deal_start_price
                                - (deal_start_price / 100.0 * total_deviation);
                            safety_order_deviation *= cfg.safety_order_step_scale;
                        }

                        bot_avg_price = bot_total_volume / bot_total_coins;
                        sell_price = bot_avg_price
                            + (bot_avg_price / 100.0 * cfg.take_profit);
                    }
                } else if high_price >= sell_price {
                    if highest_so < current_safety_order {
                        highest_so = current_safety_order;
                    }
                    total_so += current_safety_order;

                    let sell_amount = bot_total_coins * sell_price;
                    avaiable_capital += sell_amount;
                    avaiable_capital -= sell_amount / 100.0 * cfg.trading_fee;

                    let secs = (price_epoch - deal_start_epoch) as f64 / 1_000_000.0;
                    let duration = ((secs / 3600.0) * 100.0).round() / 100.0;

                    if duration > max_deal_time {
                        max_deal_time = duration;
                    }
                    deal_times.push(duration);

                    reset_data = true;
                }
            }
        }
    }

    if reset_data {
        deal_start_epoch = last_processed_epoch;
    }

    let last_epoch = last_processed_epoch;
    let last_close = last_processed_close;

    if backtest_end_epoch == 0 {
        backtest_end_epoch = last_epoch;
    }

    let secs = (cfg.end_epoch - deal_start_epoch) as f64 / 1_000_000.0;
    let duration = ((secs / 3600.0) * 100.0).round() / 100.0;

    if duration > max_deal_time {
        max_deal_time = duration;
    }
    deal_times.push(duration);

    if highest_so < current_safety_order {
        highest_so = current_safety_order;
    }
    total_so += current_safety_order;

    Results {
        avaiable_capital,
        bot_total_volume,
        bot_total_coins,
        bot_avg_price,
        sell_price,
        total_deals,
        highest_so,
        total_so,
        max_deal_time,
        backtest_start_epoch,
        backtest_end_epoch,
        last_close,
        deal_times,
    }
}
