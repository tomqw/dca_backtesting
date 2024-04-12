from functions.fetch_data import updatePriceData

# lets define the pair we want to backtest
pair = "SOLUSDT"

# update the price data for this pair.. you should comment it out to avoid fetching data every time you are testing --> put a # in front of the line.
updatePriceData(pair)
