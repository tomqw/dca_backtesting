from datetime import date
import os

from functions.bot_config import getAllConfigs, getSingleConfig, getMax
from functions.fetch_data import updatePriceData
from functions.backtest import saveResult, startBacktest

from ray import tune

from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from ray.tune.search.hebo import HEBOSearch

from ray.tune.search.ax import AxSearch

from ray.train import RunConfig

from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

import random

import statistics

# the date from which the backtest should start an end

# down, up, down, up
startDate = date(2021, 4, 14).strftime("%Y-%m-%d %H:%M:%S")
endDate = date(2024, 4, 15).strftime("%Y-%m-%d %H:%M:%S")

# all down
# startDate = date(2021, 11, 13).strftime("%Y-%m-%d %H:%M:%S")
# endDate = date(2023, 1, 10).strftime("%Y-%m-%d %H:%M:%S")

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

    # if these conditions are met skip backtest and return low score
    if config.max_amount_for_bot_usage > 1500:
        tempscore = -(config.max_amount_for_bot_usage)
        print(tempscore)
        return (
            tempscore,
            tempscore,
            round(config.max_safety_order_price_deviation, 2),
            round(config.max_amount_for_bot_usage, 2),
        )

    if config.max_amount_for_bot_usage < 130:
        tempscore = -(config.max_amount_for_bot_usage)
        print(tempscore)
        return (
            tempscore,
            tempscore,
            round(config.max_safety_order_price_deviation, 2),
            round(config.max_amount_for_bot_usage, 2),
        )

    if config.max_safety_order_price_deviation > 60:
        tempscore = -(config.max_safety_order_price_deviation * 10)
        print(tempscore)
        return (
            tempscore,
            tempscore,
            round(config.max_safety_order_price_deviation, 2),
            round(config.max_amount_for_bot_usage, 2),
        )
    
    if config.max_safety_order_price_deviation < 30:
        tempscore = (-100000 + (config.max_safety_order_price_deviation / 30) * (99000))
        print(tempscore)
        return (
            tempscore,
            tempscore,
            round(config.max_safety_order_price_deviation, 2),
            round(config.max_amount_for_bot_usage, 2),
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


# Define the search space for the optimization
# search_space = {
#     "tp": tune.quniform(0.5, 3.5, 0.1),
#     "so": tune.quniform(10, 12, 0.2),
#     "mstc": tune.randint(5, 50),
#     "sos": tune.quniform(0.5, 3.5, 0.1),
#     "os": tune.quniform(0.5, 5, 0.1),
#     "ss": tune.quniform(0.5, 5, 0.1),
# }
# Define the search space for the optimization using int and than div them
search_space_hebo = {
    "tp": tune.randint(90, 250),
    "so": tune.randint(50, 55),  # its 10 - 12 but divided by 5
    # "mstc": tune.randint(5, 31),
    "sos": tune.randint(90, 250),
    "os": tune.randint(10, 1000),
    "ss": tune.randint(10, 1000),

    #defaults:
    # "tp": tune.randint(1, 1000),
    # "so": tune.randint(50, 61),  # its 10 - 12 but divided by 5
    # # "mstc": tune.randint(11, 13),
    # "sos": tune.randint(20, 1000),
    # "os": tune.randint(10, 1000),
    # "ss": tune.randint(10, 1000),

    #with mstc 5-30 and so 10-11
    # - os should be 0.92 - 3.45, which means 180-2199 USDT
    # - ss should be 0.67 - 4.3, which means 30.3% - 93,48% deviation

    #with mstc 5 and so 10-11
    # - os should be 1.64 - 2.84, which means 137-1008 USDT
    # - ss should be 0.67 - 4.3, which means 30.3% - 93,48% deviation

    #with mstc=8 and so 10-11
    # - os should be 1.12 - 1.71, which means 146-1025 USDT
    # - ss should be 0.67 - 2.2, which means 29.07% - 95,85% deviation

    #with mstc=9 and so 10-11
    # - os should be 1.1 - 1.55, which means 146-1022 USDT
    # - ss should be 0.67 - 1.96, which means 29.47% - 93,16% deviation
    
    #with mstc=10 and so 10-11
    # - os should be 1.06 - 1.46, which means 142-1038 USDT
    # - ss should be 0.67 - 1.81, which means 29.75% - 97,58% deviation

    #with mstc=11 and so 10-11
    # - os should be 1 - 1.39, which means 120-1037 USDT
    # - ss should be 0.67 - 1.69, which means 29.93% - 97,44% deviation

    #with mstc=12 and so 10-11
    # - os should be 1 - 1.33, which means 140-1007 USDT
    # - ss should be 0.66 - 1.6, which means 29.21% - 98,16% deviation

}

search_space_ax_manual = [
    {"name": "tp", "type": "range", "bounds": [2.5, 3.3]},
    # {"name": "x2", "type": "range", "bounds": [0.0, 1.0]},
]


# Define the optimization objective
def optimize_my_function_hebo(conf):
    tp = conf["tp"] / 100
    so = conf["so"] / 5
    # mstc = conf["mstc"]
    mstc = 5
    sos = conf["sos"] / 100
    os = conf["os"] / 100
    ss = conf["ss"] / 100

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


# def optimize_my_function_ax(conf):
#     tp = conf["tp"]
#     so = 10.2
#     mstc = 11
#     sos = 3.4
#     os = 1.3
#     ss = 1


#     # getmaxtemp = so * (os**(mstc)-1) / (os - 1) + 10
#     # print(getmaxtemp)

#     # os_temp = (getmaxtemp - 10)/(so**mstc - getmaxtemp + 10)
#     # print(os_temp)

#     tempscore, tempmedian, tempdev, tempmax = runBacktest(
#         tp, so, mstc, sos, os, ss, pairs, startDate, endDate
#     )

#     tempmax = float(tempmax)  # convert from int to float to avoid error from ray tune

#     return {
#         "score": tempscore,
#         "median": tempmedian,
#         "max_safety_order_price_deviation": tempdev,
#         "max_amount_for_bot_usage": tempmax,
#     }

# TuneBOHB version

# algo = TuneBOHB()
# algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)
# scheduler = HyperBandForBOHB(
#     time_attr="training_iteration",
#     max_t=100,
#     reduction_factor=4,
#     stop_last_trials=False,
# )

# tuner = tune.Tuner(
#     optimize_my_function,
#     tune_config=tune.TuneConfig(
#         metric="score", mode="max", scheduler=scheduler, search_alg=algo, num_samples=100
#     ),
#     param_space=search_space,
# )

# HEBO version

current_best_params = [
    # {
    #     "tp": 290,
    #     "so": 51,
    #     # "mstc": 11,
    #     "sos": 340,
    #     "os": 130,
    #     "ss": 100,
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


algo_hebo = HEBOSearch(
    metric="median", mode="max", points_to_evaluate=current_best_params, max_concurrent=100
)
# algo_hebo = tune.search.ConcurrencyLimiter(algo_hebo, max_concurrent=50)

algo_hyperopt = HyperOptSearch(
    metric="median", mode="max", points_to_evaluate=current_best_params
)

algo_hyperopt = tune.search.ConcurrencyLimiter(algo_hyperopt, max_concurrent=50)

# algo_hebo.save_to_dir("c:/Users/Tomasz/ray_results/optimize_my_function_hebo_1/")

# algo_ax = AxSearch(space=search_space_ax_manual, metric="median", mode="max")

trainable_with_resources = tune.with_resources(optimize_my_function_hebo, {"cpu": 0.5})

# Disable changing the current working directory. Needed for relative/absolute paths
os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

tuner_hebo = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        search_alg=algo_hyperopt,
        num_samples=2500,
    ),
    # run_config=RunConfig(storage_path="c:/Users/Tomasz/ray_results/optimize_my_function_hebo_1/", name="test_experiment"),
    param_space=search_space_hebo,
)

# tuner_ax = tune.Tuner(
#     # optimize_my_function_hebo,
#     optimize_my_function_ax,
#     tune_config=tune.TuneConfig(
#         search_alg=algo_ax, num_samples=10, max_concurrent_trials=10,
#     )
# )

analysis = tuner_hebo.fit()

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
df.loc[:, "config/mstc"] = 5

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
