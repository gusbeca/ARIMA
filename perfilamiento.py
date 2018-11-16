import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from datetime import datetime, timedelta
from pandas import read_csv
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

#Carga de la serie y grafica en eltiempo
pathPrin= 'C:/Users/CEO-ORBECA/SynologyDrive/MAESTRIA/2do Semestre/BUSSINESS/Proyecto/1105.xlsx'
df= pd.read_excel(pathPrin,0)
df['FECHA'] =  pd.to_datetime(df['FECHA'], infer_datetime_format=True)

plt.figure(figsize=(15, 7))
plt.plot(df.FECHA,df.QNETA,'b')
plt.title('QNETA')
plt.grid(True)
plt.show()

#Carga de la serie y grafica de la funcion de autocorrelacion
def parser(x):
   return datetime.strptime(x, '%Y-%m-%d')

series = read_csv('csv1105.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series)
autocorrelation_plot(series)
plt.show()
plot_acf(series)
plt.show()

#Graficia de la funcion de autocorrelacion parcial
plot_pacf(series, lags=69)
plt.show()


df.boxplot('QNETA',by="MES")    #boxplot de continuas x categoricas
plt.title("Boxplot")
plt.suptitle("")
plt.xticks(rotation=90)
plt.show()

diff = series.diff()
plt.plot(diff)
plt.show()
print(diff.head())
plot_pacf(diff, lags=69)
plt.show()

autocorrelation_plot(diff)
plt.show()
plot_acf(diff)
plt.show()

#-- Prueba de Fuller
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries,periodo):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=periodo).mean()
    rolstd = timeseries.rolling(window=periodo).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=True)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(series,12)