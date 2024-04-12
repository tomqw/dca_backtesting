import json,sys,os,requests
from datetime import datetime,date,timedelta
from time import sleep
import time

from functions.bot_config import *
from functions.fetch_data import *
from functions.backtest import *


#lets define the pair we want to backtest
pair = 'SOLUSDT'

#update the price data for this pair.. you should comment it out to avoid fetching data every time you are testing --> put a # in front of the line.
updatePriceData(pair)
