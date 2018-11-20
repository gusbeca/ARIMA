#-------ALGO DE TEORIA:-------------------------------------------------------------------------------------------------
#---------ECONOMETRIA----------------------------------------------------------
# (EN ECONOMIETRIA UNA SERIE DE TIEMPO COMO LAS VENTAS DE UN PRODUCTO SE EXPLICA POR SUS VALORES PASADOS MAS VARIABLES
#  EXTERNAS (TABIEN LLAMADS REGRESORES) COMO: EL PRECIO ,PROMOCIONES, TEMPORARDAS, COMODITIES,.....
# LAS VARIABLES EXTERNAS PUEDEN SER DE MUCHOS TIPOS; PUEDEN SER OTRAS SERIES DE TIEMPO (HAY MODELOS ARIMA MATRICIALES
# (MULTIDIMENCIONALES) PARA ESTO) , PUEDEN SER NUMERICAS O NOMINALES
# PARA TRABAJAR SERIES DE TIEMPO DE UNA DIMENCION CON ESTACIONALIDAD (TEMPORADAS) Y CON VARAIBLES EXTERNAS  EL MODELO MAS POPULAR ES SARIMAX

#---------SERIES DE TIEMPO--------------------------------------------------------------------------------------------
# Una serie de tiempo es un proceso que puede ser modelado por una funcion del tiempo + una fucion de estado+
# un componente no terminista aleatorio o estocasico
# La serie deve ser causal es decir los valores pasados no deven depender de valores futuros.
# Si se desea hacer una prediccion la serie de tiempo NO puede ser aleatoria ni puede ser un camino aleatorio, poruqe una
#  serie aleatoria o un ramdom walk NO pueden ser predicho.
#(Hay que validar estos supuestos antes de empzar a hacer cualquier modelo)
# El objetivo de la prediccion en series de tiempo es encontrar los patrones o las fucniones deterministas quedando
#  solo la parte aletoria o ruido blanco
# #MAXIMA: Una serie de tiempo puramente aleatorioa "ruido blanco" NO PUEDE ser pronosticada
# Una serie de tiempo es la realizacion de una sucecion de eventos aleatorios. "El problema del camino aleatorio"
# Un modelo de prediccion de series de tiempo debe ser validado contra un modelo ingenuo como: "mañana va a ser igual que ayer",
# de otra manera puede generar modelos que parace estar prediciondo pero no predicen nada solo son un desface temporal.
# Por ejemplo un modelo aplicado a ruido blanco puede parecer predecir la serie pero al verificar contra el modelo ingenuo se ve que hace nada.

# LOS METODOS MAS POPULARES (EN COMPENTENCIAS DE ML) PARA PREDECIR SERIES DE TIEMPO UNIDIMENCIONALES SON SARIMAX Y THETA FORECASTING Y GRADIENTBOOSTEDTREES

#----- EL ALGORITMO SARIMAX------------------------------------------------------------------------------------------------------

#DESCOMPONIENDO LA SIGLA SARIMAX:  AUTOREGRESIVO (AR)+ MEDIAS MOVILES (MA) + UNA PARTE DIFRENREICAL (I)
# + UNA PARATE PARA MANEJO DE ESTACINALIDAD (S) +  UNA PARTE PARA MAJEO DE VARIABLES EXTERNAS X

#(AR) O AUTO REGRESIVO ES UN MODELO DONDE LOS VALORES FUTUROS DE LA SERIE SE EXPLICAN COMO UNA CONVINACION LINEAL
# O PONDERADCION DE VALORES PASADOS MAS UNA COMPOINENTE ALEATORIA
# EL SUPUESTO DE AR ES QUE LA SERIE ES INVERTIBLE LO QUE QUIERE DECIR QUE LOS VALORES RECIENTES SON MAS RELEVANTES
# QUE LOS ANTIGUOS PARA LA PREDICCION DEL VALOR FUTURO.

#(MA) O MEDIA MOVILES ES UN MODELO DONDE LOS VALORES FUTUROS DE LA SERIE SE EXPLICAN COMO UNA COMBINACION LINEAL DE LOS
#  ERRORES DE LAS PREDICCIONES PASADAS.
# EL SUPUESTO DE MA ES QUE LA SERIE ES ESTACIONARIA, ESTACIONARIA SIGNIFICA QUE SU MEDIA Y VARAINZA NO CAMBIAN CON EL
# TIEMPO Y QUE LA COVARIANZA CON CADA VALOR PASADO ES COSTANTE
# EN TERMINOS DE INGENIRO ELECTRONICO SIGNIFICA QUE LA SEÑAL TIENE AMPLITUD CONSTANTE, NIVEL DC CONSTANTE Y QUE SU
# FRECUENCIA ES CONSTANTE
# IMPORTANTE: (ESE COSTANTE ES RELATIVO A LA VENTANA DE TIEMPO CONQUE SE MIRE)

#(I) INTEGRAL, SIMPLEMENTE ES UNA TRASFORMACION DE LA SERIE PARA CUMPLIR CON EL SUPEUSTO DE ESTACIONRAIDAD LO QUE HACE ES
#  DIFERENCIAR LA SERIE DE TIEMPO
#IMPORTANTE DIFERENCIA LA SERIE DE TIEMPO NO ASEGURA ESTACIONARIDAD SE DEBE VERIFICAR, DE FORMA GRAFICA Y O CON LA PRUEBA
# DE DICKE FULLER
# LAS TRASNFORMACIONES MAS COMUNES PARA LOGRAR ESTACIONARIDAD SON LA DIFERNECIACION (TIPICAMENTE ARREGLA LA MEDIA),
# LA LOGARITMACION (TIPICAMENTE ARREGLA LA VARIANZA) O LA DIFERENCIACION RELATIVA( TIPICAMENTE ARREGLA AMBAS)
# SE PUEDE REQUERIR DE VARIAS TRASFORMACIONES SUSECIVAS

#(S) ES UNA GENERALIZACION DE ARIMA, AÑADE UNA PARTE AL AGLORITMO QUE MANERA LA ESTACIONALIDAD.

#(X) ES UN AGENRALIZACION DE SARIMA, LA X ES PORGE AGEEGA UNA RERESION LINEAL CON VARIABLES EXTERNAS

#SARIMAX(p,d,q)(P,D,Q), p es la cantida de valores pasados usados en AR, d es el numero de veces que se diferencia la serie,
# q es el  numero de errores de predicciones pasadas unsado en MA, P,D y Q son lo mismo pero para la partes estacional.

#SUPUESTOS DE ARMA: SERIE ESTACIONARIA, 2 SERIE INVERTIBLE, SERIE NO ESTACIONAL (NO HAY TEMPORADAS CICLOS)

#*Nota: Estacionaridad (estacionary) != de estacionalidad (seasonal).
#----------------------------------------------------------------------------------------------------------------------

#----------------IMPROTACION DE LIBRERIAS------------------------------------------------------------------------------
import pandas as pd
from pandas import read_csv
import plotly
from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas import read_csv
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from scipy.fftpack import fft
#-----------------INICIALIZACIONES-------------------------------------------------------------------------------------
plotly.tools.set_credentials_file(username='busongeneral86', api_key='bRnpgRBZYPhYnLXNQfvH')
plotly.tools.set_config_file(world_readable=True,
                             sharing='public')

#------------------CARGA DE DATOS--------------------------------------------------------------------------------------

def parser(x):
   return datetime.strptime(x, '%Y-%m-%d')
data = read_csv('csv1105.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

#----------------EXPLORACION DE DATOS---------------------------------------------------------------------------------
#Supuestos: NO es un proceso aleatorio,  NO es un camino aletorio.
#Si es camino aleatorio no hay meandros, si es  un camino laerorio aleatorio las difenrecias seran un ruido blanco
# Si es aletorio la varainza de als diferencias sera menor que la varianza de la serie original
#Si es un ramdom walk o caminata de borracho la funciond e correlacion sera muy alta para el primier valor anterior y descendera lienalmente para el resto



#Si no es un proceso aleatorio
# Se hace la descomposicion de la serie de tiempo para er las componetes:
# tendencia y ciclica (fluctuaciones repetitivas per NO periodicas),
#  periodica "estacional" y
#  ruido "irregular"
# El odelo de descomposicion puede ser aditivo y_{t}=T_{t}+C_{t}+S_{t}+I_{t} o multiplicativo y_{t}=T_{t}xC_{t}xS_{t}xI_{t}
# El multiplicativo se usa cuando la tendencia no es linea ejemplo: exponencial
#Carga de la serie y grafica en eltiempo
pathPrin= '1105.xlsx'
df= pd.read_excel(pathPrin,0)
df['FECHA'] =  pd.to_datetime(df['FECHA'], infer_datetime_format=True)

df.boxplot('QNETA',by="MES")    #boxplot de continuas x categoricas
plt.title("Boxplot")
plt.suptitle("")
plt.xticks(rotation=90)
plt.show()
#La grafica mujestra estacionalidad (La distribucion cambia segun el mes)

#---- TRANSFORMADA DE FOURIER-- COMO ELECTRONICO NO DEBE FALTAR ,
# WAVELET TAMBIE N SE USA MUCHO, AL IGUAL QUE FILTROS PASA BAJO (SUAVIAZADO O smoothing O MEDIAS MOVLES)
#FFT
#number of sample points
N=data.shape[0]
#frequency of signal
T = 1/12
#create x-axis for time length of signal
x = np.linspace(0, N*T, N)
# Quitando el nivel DC
y=data-data.mean()
#perform FFT on signal
yf = fft(y)
#create new x-axis: frequency from signal
xf = np.linspace(0, 1.0/(2.0*T), N//2)
#plot results
plt.plot(xf, abs(yf[0:N//2]))
plt.grid()
plt.title("FFT")
plt.show()
print(yf)

#--- FUNCION DE AUTOCORRELACION Y FUNCION DE AUTOCORRELACION PARCIAL---
# ES COMO UNA FFT PERO EN LUGAR DE COMPONENTES EN FRECUENCIA SON COMPONENTES EN VALORES PASADOS
autocorrelation_plot(data)
plt.show()
plot_acf(data)
plt.show()

#Esta grafica sirrve para:
# Verificar que nosea una serie totalmetne aletoria la autocorrelacion de una funciona leatoria es 0 o esta por debajo del intervalo de confiza
# Indicaciones sobre si es un ramdum walk


#Graficia de la funcion de autocorrelacion parcial
plot_pacf(data)
plt.show()

#----------DESCOMPOSICION-----------------------------------------------------------------------
result = seasonal_decompose(data, model='additive')
#fig = result.plot()
#plot_mpl(fig)
#La grafica muestra el componente estacional y  una clara tendencia lineal creciente lo que implica NO estacionaridad

#---------VERIFICACION DE SUPUESTOS: ESTACIONALIDAD-----------------------------------------------------------------
#-- Prueba de Dickey Fuller y grafica de mediva y desviacion estandar moviles.

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

test_stationarity(data,12)


#----------------TRASNFORMACION DE DATOS-------------------------------------------------------------------------------
#---TRASNFORMACIONES PARA CUMPLIR CON EL SUPUESTO DE SERIE ESTACIONARIA............................................
diffr=data.rolling(2).apply(lambda x: (x[0]-x[1])/x[1],raw=True) # DIFERENCIA RELATIA PARA REDUCIR LA VARIACION TEMPORAL DE LA MEDIA Y LA VARAINZA
diffr=diffr.dropna()
y=diffr
#--- CON ESTO LA MEDIA SE HACE APRO CONSTANTE  PERO LA VARAINZA SIGUE DEPENDIENDO DEL TIEMPO

diffrdiff=diffr.diff()
diffrdiff= diffrdiff.dropna()
dy=diffrdiff
# CON LA SEGUNDA DIFERENCIACION YA LA SERIE PAS ALA PRUEBA DE Dickey-Fuller






from pmdarima.arima import auto_arima
stepwise_model = auto_arima(y, start_p=0, start_q=0,
                           #max_d= 2, max_D = 2,
                           max_p=12, max_q=12, m=12,
                           start_P=0, seasonal=True,
                           trace=True, d=1, D=0,
                           error_action='ignore',
                           suppress_warnings=True, stationary=False,
                           aproximation=False,#Para usras aproximaciones y reducir tiempo de ejecucion
                           stepwise=False, #eJECUCION PASO A PASO RAPIDA
                           parallel=True)
print(stepwise_model.aic())
print('acabo esta parte')
import datetime as dt
start = dt.datetime.strptime('2013-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2018-01-01', '%Y-%m-%d')
print('acabo esta parte2')
train= y.loc[(data.index > start) & (data.index < end)]
print('acabo esta parte3', len(train))
test = y.iloc[len(train):]
print('acabo esta parte4')
#print('Entrenameinto '+len(train)+'test: '+len(test))
stepwise_model.fit(train)
print('acabo esta parte5')
future_forecast = stepwise_model.predict(n_periods=9)
print(future_forecast)

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
pd.concat([test,future_forecast],axis=1).plot()

pd.concat([data,future_forecast],axis=1).plot()