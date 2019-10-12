import csv
import json
import requests

apiKey = "O4KE97KEIMO74MNW"
symbol = "FCX"
filepath = symbol+".csv"

rsiTimePeriod = str(14)
smaTimePeriod = str(50)
emaTimePeriod = str(13)
bbandsTimePeriod = str(20)

rsiSeriesType = "close"
smaSeriesType = "close"
emaSeriesType = "close"
bbandsSeriesType = "close"
nbdevup = str(2)
nbdevdn = str(2)

OHLC_res = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol="+symbol+"&outputsize=full&apikey="+apiKey+"&datatype=csv")
RSI_res = requests.get("https://www.alphavantage.co/query?function=RSI&symbol="+symbol+"&interval=daily&time_period="+rsiTimePeriod+"&series_type="+rsiSeriesType+"&apikey="+apiKey+"&datatype=csv")
SMA_res = requests.get("https://www.alphavantage.co/query?function=SMA&symbol="+symbol+"&interval=daily&time_period="+smaTimePeriod+"&series_type="+smaSeriesType+"&apikey="+apiKey+"&datatype=csv")
EMA_res = requests.get("https://www.alphavantage.co/query?function=EMA&symbol="+symbol+"&interval=daily&time_period="+emaTimePeriod+"&series_type="+emaSeriesType+"&apikey="+apiKey+"&datatype=csv")
BBANDS_res = requests.get("https://www.alphavantage.co/query?function=BBANDS&symbol="+symbol+"&interval=daily&time_period="+bbandsTimePeriod+"&series_type="+bbandsSeriesType+"&nbdevup="+nbdevup+"&nbdevdn="+nbdevdn+"&apikey="+apiKey+"&datatype=csv")

csvfile = open(filepath,"w",newline='')

OHLCstringlist = (OHLC_res.text).split("\r\n")
RSIstringlist = (RSI_res.text).split("\r\n")
SMAstringlist = (SMA_res.text).split("\r\n")
EMAstringlist = (EMA_res.text).split("\r\n")
BBANDSstringlist = (BBANDS_res.text).split("\r\n")

print("OHLC Len: "+str(len(OHLCstringlist)))
print("RSI Len: "+str(len(RSIstringlist)))
print("SMA Len: "+str(len(SMAstringlist)))
print("EMA Len: "+str(len(EMAstringlist)))
print("BBANDS Len: "+str(len(BBANDSstringlist)))

print(OHLC_res)
print(RSI_res)
print(SMA_res)
print(EMA_res)
print(BBANDS_res)

csvwriter = csv.writer(csvfile, delimiter=",")
count = 0   #first index is the most recent data point 
while count<(len(OHLCstringlist)-int(smaTimePeriod)-1):
    csvwriter.writerow(OHLCstringlist[count].split(",")+[RSIstringlist[count].split(",")[1]]+[SMAstringlist[count].split(",")[1]]+[EMAstringlist[count].split(",")[1]]+BBANDSstringlist[count].split(",")[1:])
    count +=1
