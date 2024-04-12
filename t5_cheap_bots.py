from datetime import date

from functions.bot_config import getConfigsByMaxBotUsage
from functions.fetch_data import updatePriceData
from functions.backtest import saveResult, startBacktest

#lets define the pair we want to backtest
pair = 'SOLUSDT'

#update the price data for this pair.. you should comment it out to avoid fetching data every time you are testing --> put a # in front of the line.
updatePriceData(pair)

#the date from which the backtest should start
startDate = date(2022,4,1).strftime("%Y-%m-%d %H:%M:%S")
#this date can be different from the initialStartDate we used to fetch our data.
#so we can download all data from the last 6 months but we start our backtest 3 months ago.. or 3 days.. whatever
#we will format this date as a string and it should look like this 2021-11-01 00:00:00
#the format found in the price files ('SOLUSDT.txt') needs to be the same format!!!

#the date at which the backtest should end. by default the test runs as long as data is avaiable..
endDate = date(2022,4,30).strftime("%Y-%m-%d %H:%M:%S")

#we can use getConfigsByMaxBotUsage to test all bots that have a max_amount_for_bot_usage between min and max
#this returns all bots that are less then or equal to 350$
config_list = getConfigsByMaxBotUsage(0,350)

whitelist=[]
results=[]
for config in config_list:
    if config.config_name in whitelist or whitelist==[]:
        result = startBacktest(config,pair,startDate,endDate)
        #lets say we only want to find the configs with a profit of at least 5 percent
        #if result['profit_percent'] > 5:
            #print (result['config_name'],result['profit'],result['max_safety_order_price_deviation'],result['max_amount_for_bot_usage'],result['profit_percent'])
        results.append(result)
        print (config.config_name, result['profit'],result['max_amount_for_bot_usage'],result['profit_percent'])
#we can save the result to a csv file for further analysis (folder = results)
saveResult(results,'cheap_bots.csv')
