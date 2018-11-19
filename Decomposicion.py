import pandas as pd
from pandas import read_csv
import plotly
plotly.tools.set_credentials_file(username='busongeneral86', api_key='bRnpgRBZYPhYnLXNQfvH')
plotly.tools.set_config_file(world_readable=True,
                             sharing='public')
from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
def parser(x):
   return datetime.strptime(x, '%Y-%m-%d')
data = read_csv('csv1105.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

diffr=data.rolling(2).apply(lambda x: (x[0]-x[1])/x[1])
diffr=diffr.dropna()
y=diffr
diffrdiff=diffr.diff()
diffrdiff= diffrdiff.dropna()
dy=diffrdiff

#print(data.head(30))
result = seasonal_decompose(y, model='additive')
fig = result.plot()
#plot_mpl(fig)



from pmdarima.arima import auto_arima
stepwise_model = auto_arima(y, start_p=0, start_q=0,
                           max_d= 2, max_D = 2,
                           max_p=12, max_q=12, m=12,
                           start_P=0, seasonal=True,
                           trace=True, d=1, D=1,
                           error_action='ignore',
                           suppress_warnings=True, stationary=False,
                           aproximation=False,#Para usras aproximaciones y reducir tiempo de ejecucion
                           stepwise=False, #eJECUCION PASO A PASO RAPIDA
                           parallel=True)
print(stepwise_model.aic())

