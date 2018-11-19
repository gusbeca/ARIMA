import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from datetime import datetime, timedelta
from pandas import read_csv
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

#---- PRIMERA PARTE EXPLORACION-------------------------------------------------------------------------------------

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


from scipy.fftpack import fft


#FFT
#number of sample points
N=series.shape[0]
#frequency of signal
T = 1/12
#create x-axis for time length of signal
x = np.linspace(0, N*T, N)
print(series.mean())
y=series-series.mean()
#perform FFT on signal
yf = fft(y)
#create new x-axis: frequency from signal
xf = np.linspace(0, 1.0/(2.0*T), N//2)
#plot results
plt.plot(xf, abs(yf[0:N//2]))
plt.grid()
plt.show()
print(yf)

#---------SEGUNDA PARTE TRASNFORMACION PARA CUMPLIR CON SUPUESTOS------------------------------------------------------
diffr=series.rolling(2).apply(lambda x: (x[0]-x[1])/x[1])
diffr=diffr.dropna()
test_stationarity(diffr,12)
diffrdiff=diffr.diff()
diffrdiff= diffrdiff.dropna()
test_stationarity(diffrdiff,12)

y=diffrdiff

autocorrelation_plot(y)
plt.show()
plot_acf(y)
plt.show()

#Graficia de la funcion de autocorrelacion parcial
plot_pacf(series, lags=69)
plt.show()

#----------TERCERA PARTE ELABORACION DEL MODELO-------------------------------------------------------------------------

import warnings
import itertools
import statsmodels.api as sm

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 5)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


#---CODIGO PARA SELECCIONAR PARAMETROS SARIMAX
warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue