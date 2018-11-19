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
print(data.head(30))
result = seasonal_decompose(data, model='multiplicative')
fig = result.plot()
plot_mpl(fig)

from pyramid.arima import auto_arima
stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())