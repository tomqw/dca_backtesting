import math
import os
import statistics
from datetime import date

import optuna
import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune.search.optuna import OptunaSearch
from scipy.optimize import fsolve

from functions.backtest import startBacktest
from functions.bot_config import getMax

# Bot settings
search_space_min_bot_usage = 750
search_space_max_bot_usage = 3000
search_space_min_bot_dev = 61
search_space_max_bot_dev = 100
search_space_base_order = 10
search_space_max_safety_orders_min = 45
search_space_max_safety_orders_max = 61
search_space_tp_min = 90
search_space_tp_max = 400
search_space_so_min = 50
search_space_so_max = 90
search_space_sos_min = 100
search_space_sos_max = 400

# The date from which the backtest should start and end

# down, up, down, up
# startDate = date(2021, 4, 14).strftime("%Y-%m-%d %H:%M:%S")
# endDate = date(2024, 4, 15).strftime("%Y-%m-%d %H:%M:%S")


# all down
startDate = date(2021, 11, 13).strftime("%Y-%m-%d %H:%M:%S")
endDate = date(2023, 1, 10).strftime("%Y-%m-%d %H:%M:%S")


# RAY settings
os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "50"
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
# os.environ["RAY_TEMP_DIR"] = "D:\\ray_tmp" # Choose a short path

ray_num_cpus_to_use = 28
ray_num_samples = 10
ray_max_concurrent_trials = 15
ray_optuna_startup_trials = 50


if not ray.is_initialized():
    print(f"Initializing Ray with num_cpus={ray_num_cpus_to_use}")
    try:
        ray.init(
            num_cpus=ray_num_cpus_to_use,
            include_dashboard=False,  # Explicitly request the dashboard
            # dashboard_host="0.0.0.0", # Makes it accessible from other devices on your network if needed, 127.0.0.1 for local only
            # dashboard_port=8265, # Default, but you can try changing if you suspect a conflict
            ignore_reinit_error=True,
        )
    except Exception as e:
        print(f"Error during Ray initialization: {e}")
        # Potentially try without dashboard if it's critical path for script to run
        # ray.init(num_cpus=num_cpus_to_use, include_dashboard=False, ignore_reinit_error=True)

print(f"Ray Cluster Resources: {ray.cluster_resources()}")
print(f"Ray Available Resources (after init): {ray.available_resources()}")


# we can also test one pair against all configs.


# we are overwriting pairs here.. pairs is defined in fetch_data.. but in this example we want only two pairs..
pairs = [
    "AAVEUSDT",
    "ADAUSDT",
    "ARUSDT",
    "ATOMUSDT",
    "AVAXUSDT",
    "BTCUSDT",
    "DOTUSDT",
    "ENJUSDT",
    "EOSUSDT",
    "ETHUSDT",
    "FILUSDT",
    "FTMUSDT",
    "LINKUSDT",
    "LUNAUSDT",
    "MATICUSDT",
    "NEOUSDT",
    "ONEUSDT",
    "SANDUSDT",
    "SOLUSDT",
    "THETAUSDT",
    "UNIUSDT",
    "VETUSDT",
    "WAVESUSDT",
    "XLMUSDT",
    "XRPUSDT",
    "XTZUSDT",
    "ZILUSDT",
]
# for pair in pairs:
# updatePriceData(pair)


def runBacktest(tp, so, mstc, sos, os, ss, pairs, startDate, endDate):
    # config = getSingleConfig("euphoria")

    class config:
        def __init__(self):
            self.config_name = "scarta+"
            self.take_profit = 2.0
            self.base_order = 10
            self.safety_order = 10
            self.max_safety_orders = 31
            self.deviation_to_open_safety_order = 2.0
            self.safety_order_volume_scale = 1.08
            self.safety_order_step_scale = 0.97

    config.config_name = "scarta+"
    config.take_profit = tp
    config.base_order = 10
    config.safety_order = so
    config.max_safety_orders = mstc
    config.deviation_to_open_safety_order = sos
    config.safety_order_volume_scale = os
    config.safety_order_step_scale = ss
    config.max_safety_order_price_deviation, config.max_amount_for_bot_usage = getMax(
        config
    )

    results = []
    for pair in pairs:
        result = startBacktest(config, pair, startDate, endDate)
        results.append(result)
        # print(
        #     config.config_name,
        #     pair,
        #     result["profit"],
        #     result["max_amount_for_bot_usage"],
        #     result["profit_percent"],
        # )

    # average profit:
    average_profit_percent = sum(item["profit_percent"] for item in results) / len(
        results
    )

    # median profit:
    median_profit_percent = statistics.median(
        [item["profit_percent"] for item in results]
    )

    # print(round(config.max_safety_order_price_deviation))
    # print(round(config.max_amount_for_bot_usage))

    return (
        average_profit_percent,
        median_profit_percent,
        round(config.max_safety_order_price_deviation, 2),
        round(config.max_amount_for_bot_usage, 2),
    )


# print(runBacktest(10, 10, pairs, startDate, endDate))


# functions for narrowing search space
def calculate_max_safety_orders_from_amount(
    max_amount_for_bot_usage, base_order, safety_order, safety_order_volume_scale
):
    if safety_order_volume_scale == 1:
        if safety_order == 0:
            raise ValueError(
                "safety_order cannot be zero when safety_order_volume_scale is 1"
            )
        max_safety_orders = (max_amount_for_bot_usage - base_order) / safety_order
    else:
        if safety_order_volume_scale <= 0:
            raise ValueError("safety_order_volume_scale must be positive")
        arg = (
            1
            + (max_amount_for_bot_usage - base_order)
            * (safety_order_volume_scale - 1)
            / safety_order
        )
        if arg <= 0:
            raise ValueError("Invalid parameters: logarithm argument must be positive")
        max_safety_orders = math.log(arg, safety_order_volume_scale)
    return int(round(max_safety_orders))


def calculate_max_safety_orders_from_deviation(
    safety_order_step_scale,
    base_order,
    safety_order,
    max_safety_order_price_deviation,
    deviation_to_open_safety_order,
):
    if safety_order_step_scale != 1:
        # Rearrange the formula to solve for max_safety_orders using the math library
        max_safety_orders = math.log(
            1
            - (max_safety_order_price_deviation / deviation_to_open_safety_order)
            * (1 - safety_order_step_scale),
        ) / math.log(safety_order_step_scale)
    else:
        # If safety_order_step_scale is 1, the formula simplifies to this
        max_safety_orders = (
            max_safety_order_price_deviation / deviation_to_open_safety_order
        )

    return int(round(max_safety_orders))


def calculate_safety_order_step_scale(
    max_safety_orders,
    deviation_to_open_safety_order,
    max_safety_order_price_deviation,
    initial_guess=1.1,
):
    """
    Calculate the safety order step scale given other parameters using fsolve.

    Parameters:
    - max_safety_orders: Maximum number of safety orders (int or float)
    - deviation_to_open_safety_order: Initial deviation to open safety order (float)
    - max_safety_order_price_deviation: Target maximum safety order price deviation (float)
    - initial_guess: Initial guess for the safety order step scale (float, default=1.1)

    Returns:
    - Rounded value of the calculated safety order step scale
    """

    def equation(safety_order_step_scale):
        if safety_order_step_scale == 1:
            return (
                deviation_to_open_safety_order * max_safety_orders
                - max_safety_order_price_deviation
            )
        else:
            return (
                deviation_to_open_safety_order
                * (1 - safety_order_step_scale**max_safety_orders)
                / (1 - safety_order_step_scale)
            ) - max_safety_order_price_deviation

    # Use fsolve to find the root of the equation
    solution = fsolve(equation, initial_guess)
    return round(solution[0], 2)


def calculate_safety_order_volume_scale(
    base_order,
    safety_order,
    max_safety_orders,
    max_amount_for_bot_usage,
    initial_guess=1.1,
):
    """
    Calculate the safety order volume scale given other parameters using fsolve.

    Parameters:
    - base_order: Base order amount (float)
    - safety_order: Safety order amount (float)
    - max_safety_orders: Maximum number of safety orders (int or float)
    - max_amount_for_bot_usage: Target maximum amount for bot usage (float)
    - initial_guess: Initial guess for the safety order volume scale (float, default=1.1)

    Returns:
    - Rounded value of the calculated safety order volume scale
    """

    def equation(safety_order_volume_scale):
        if safety_order_volume_scale == 1:
            return (
                base_order + safety_order * max_safety_orders - max_amount_for_bot_usage
            )
        else:
            return (
                base_order
                + safety_order
                * (1 - safety_order_volume_scale**max_safety_orders)
                / (1 - safety_order_volume_scale)
            ) - max_amount_for_bot_usage

    # Use fsolve to find the root of the equation
    solution = fsolve(equation, initial_guess)
    return round(solution[0], 2)


def pythonic_search_space(trial):
    mstc = trial.suggest_int(
        "mstc", search_space_max_safety_orders_min, search_space_max_safety_orders_max
    )
    tp = trial.suggest_int("tp", search_space_tp_min, search_space_tp_max)
    so = trial.suggest_int("so", search_space_so_min, search_space_so_max)
    sos = trial.suggest_int("sos", search_space_sos_min, search_space_sos_max)
    ss_min = (
        calculate_safety_order_step_scale(
            mstc, sos / 100, search_space_min_bot_dev, initial_guess=1.1
        )
        * 100
    )
    ss_max = (
        calculate_safety_order_step_scale(
            mstc, sos / 100, search_space_max_bot_dev, initial_guess=1.1
        )
        * 100
    )
    os_min = (
        calculate_safety_order_volume_scale(
            search_space_base_order,
            so / 5,
            mstc,
            search_space_min_bot_usage,
            initial_guess=1.1,
        )
        * 100
    )
    os_max = (
        calculate_safety_order_volume_scale(
            search_space_base_order,
            so / 5,
            mstc,
            search_space_max_bot_usage,
            initial_guess=1.1,
        )
        * 100
    )
    os = trial.suggest_int("os", os_min, os_max)
    ss = trial.suggest_int("ss", ss_min, ss_max)


# Define the optimization objective
def optimize_my_function(conf):
    tp = conf["tp"] / 100
    so = conf["so"] / 5
    sos = conf["sos"] / 100
    os = conf["os"] / 100
    ss = conf["ss"] / 100
    mstc = conf["mstc"]

    # getmaxtemp = so * (os**(mstc)-1) / (os - 1) + 10
    # print(getmaxtemp)

    # os_temp = (getmaxtemp - 10)/(so**mstc - getmaxtemp + 10)
    # print(os_temp)

    tempscore, tempmedian, tempdev, tempmax = runBacktest(
        tp, so, mstc, sos, os, ss, pairs, startDate, endDate
    )

    tempmax = float(tempmax)  # convert from int to float to avoid error from ray tune

    return {
        "score": tempscore,
        "median": tempmedian,
        "max_safety_order_price_deviation": tempdev,
        "max_amount_for_bot_usage": tempmax,
    }


current_best_params = [
    # {
    #     # sliderb
    #     "tp": 169,
    #     "so": 71,
    #     "mstc": 13,
    #     "sos": 142,
    #     "os": 140,
    #     "ss": 119,
    # },
    # {
    #     "tp": 290,
    #     "so": 50,
    #     # "mstc": 11,
    #     "sos": 340,
    #     "os": 130,
    #     "ss": 100,
    # },
    # {
    #     "tp": 437,
    #     "so": 51,
    #     # "mstc": 11,
    #     "sos": 627,
    #     "os": 131,
    #     "ss": 81,
    # },
    # {
    #     "tp": 449,
    #     "so": 54,
    #     # "mstc": 11,
    #     "sos": 664,
    #     "os": 132,
    #     "ss": 81,
    # },
]


optuna_sampler = optuna.samplers.TPESampler(n_startup_trials=ray_optuna_startup_trials)

algo = OptunaSearch(
    pythonic_search_space,
    metric="median",
    mode="max",
    points_to_evaluate=current_best_params,
    sampler=optuna_sampler,  # Set the number of startup trials here
)


algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=ray_max_concurrent_trials)


trainable_with_resources = tune.with_resources(optimize_my_function, {"cpu": 1})

# Disable changing the current working directory. Needed for relative/absolute paths


tuner = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        search_alg=algo,
        num_samples=ray_num_samples,
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",  # shorten log directory name
    ),
    run_config=RunConfig(
        # storage_path="d:/ray",
        # name="test_experiment",
        # minimal logging
        log_to_file=False,
        verbose=1,
    ),
    # param_space=search_space,
)


analysis = tuner.fit()

# Print the best result
# print("Best result:", analysis.best_result)
print(analysis.get_best_result(metric="median", mode="max").config)

df = analysis.get_dataframe()

print(df)

print(df["config/so"])

df.sort_values(by="median", ascending=False, inplace=True)

print(df.head(20))

# df.drop(columns=["score", "timestamp", "checkpoint_dir_name", "done", "training_iteration","date"], inplace=True)

# add missing columns
df["INDEX"] = df.index + 1
df.loc[:, "creator"] = "Tom"
df.loc[:, "Deal Start Condition"] = "ASAP"
df.loc[:, "(BO) Base Order Size"] = 10
# df.loc[:, "config/mstc"] = 5

# divide by 10 because they were multiplied by 10

df["config/tp"] = df["config/tp"] / 100
df["config/so"] = df["config/so"] / 5
df["config/os"] = df["config/os"] / 100
df["config/ss"] = df["config/ss"] / 100
df["config/sos"] = df["config/sos"] / 100

df.to_csv("results/wynik2.csv", sep=";")

# list of columns to be exported to csv
columns_to_keep = [
    "INDEX",
    "creator",
    "trial_id",
    "config/tp",
    "(BO) Base Order Size",
    "config/so",
    "config/os",
    "config/ss",
    "config/sos",
    "config/mstc",
    "Deal Start Condition",
    "max_safety_order_price_deviation",
    "max_amount_for_bot_usage",
]

print(df["config/so"])

# export dataframe to csv
df[columns_to_keep].to_csv(
    "results/output.csv",
    index=False,
    header=[
        "INDEX",
        "Creator",
        "What's the name of the Bot Config?",
        "(TP) Take Profit",
        "(BO) Base Order Size",
        "(SO) Safety Order Size",
        "(OS) Safety Order Volume Scale",
        "(SS) Safety Order Step Scale",
        "(SOS) Price Deviation to Open Safety Orders",
        "(MSTC) Max Safety Trades Count",
        "Deal Start Condition",
        "max_safety_order_price_deviation",
        "max_amount_for_bot_usage",
    ],
)
