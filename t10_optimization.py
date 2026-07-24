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
from functions.config import get_pairs

# Bot settings
search_space_min_bot_usage = 750
search_space_max_bot_usage = 3000
search_space_min_bot_dev = 61
search_space_max_bot_dev = 100
search_space_base_order = 10
search_space_max_safety_orders_min = 45
search_space_max_safety_orders_max = 61
search_space_tp_min = 0.9
search_space_tp_max = 4.0
search_space_so_min = 10.0
search_space_so_max = 18.0
search_space_sos_min = 1.0
search_space_sos_max = 4.0

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


# we are using pairs from config
pairs = get_pairs()


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
def calculate_safety_order_step_scale(
    max_safety_orders,
    deviation_to_open_safety_order,
    max_safety_order_price_deviation,
    initial_guess=1.1,
):
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
    return solution[0]


def calculate_safety_order_volume_scale(
    base_order,
    safety_order,
    max_safety_orders,
    max_amount_for_bot_usage,
    initial_guess=1.1,
):
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
    return solution[0]


def pythonic_search_space(trial):
    mstc = trial.suggest_int(
        "mstc", search_space_max_safety_orders_min, search_space_max_safety_orders_max
    )
    tp = trial.suggest_float("tp", search_space_tp_min, search_space_tp_max, step=0.01)
    so = trial.suggest_float("so", search_space_so_min, search_space_so_max, step=0.2)
    sos = trial.suggest_float("sos", search_space_sos_min, search_space_sos_max, step=0.01)
    ss_min = round(
        calculate_safety_order_step_scale(
            mstc, sos, search_space_min_bot_dev, initial_guess=1.1
        )
        + 0.005,
        2,
    )
    ss_max = round(
        calculate_safety_order_step_scale(
            mstc, sos, search_space_max_bot_dev, initial_guess=1.1
        ),
        2,
    )
    os_min = round(
        calculate_safety_order_volume_scale(
            search_space_base_order,
            so,
            mstc,
            search_space_min_bot_usage,
            initial_guess=1.1,
        ),
        2,
    )
    os_max = round(
        calculate_safety_order_volume_scale(
            search_space_base_order,
            so,
            mstc,
            search_space_max_bot_usage,
            initial_guess=1.1,
        ),
        2,
    )
    os = trial.suggest_float("os", os_min, os_max, step=0.01)
    ss = trial.suggest_float("ss", ss_min, ss_max, step=0.01)


# Define the optimization objective
def optimize_my_function(conf):
    tp = conf["tp"]
    so = conf["so"]
    sos = conf["sos"]
    os = conf["os"]
    ss = conf["ss"]
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

# values already in correct float range from suggest_float

float_cols = ["config/tp", "config/so", "config/sos", "config/os", "config/ss"]
df[float_cols] = df[float_cols].round(2)

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
