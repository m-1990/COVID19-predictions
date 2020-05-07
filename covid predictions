import pandas as pd
import numpy as np
import datetime

import folium 
from folium import plugins

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import *

from opencage.geocoder import OpenCageGeocode

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings

%config InlineBackend.figure_format = 'retina'

warnings.filterwarnings('ignore')
%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
confirmed = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')
recovered = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv')
deaths= pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv')
test_data= pd.read_csv('/kaggle/input/covid19-challenges/test_data_canada.csv')
test_data.head()
recovered_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_recovered.csv')
recovered_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_recovered.csv')
study = recovered_df.loc[recovered_df['date']=='2020-04-03',['province','cumulative_recovered']]
study.index = study['province']
study.drop('province',axis=1,inplace=True)
study
death_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_mortality.csv')
test_intl=pd.read_csv('/kaggle/input/covid19-challenges/test_data_intl.csv')
test_intl.head()
intl_death= pd.read_csv('/kaggle/input/covid19-challenges/international_mortality.csv')

f_column = intl_death["deaths"]
intl_death.tail()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

case_vs_recovered = pd.concat([test_data['province'].value_counts(),study,death_df['province'].value_counts()],axis=1,sort=False)
case_vs_recovered.index.name='province'
case_vs_recovered.columns = ['Confirmed','Recovered','Death']


case_vs_recovered.fillna(0,inplace=True)
case_vs_recovered = case_vs_recovered.astype(int)


display(case_vs_recovered)

recover_rate = pd.DataFrame([elem + "%" if elem!="nan" else "0%" for elem in map(str,round(case_vs_recovered['Recovered'] / case_vs_recovered['Confirmed'] * 100,2))],index=case_vs_recovered.index,columns=['Recover Rate(%)'])
death_rate = pd.DataFrame([elem + "%" if elem!="nan" else "0%" for elem in map(str,round(case_vs_recovered['Death'] / case_vs_recovered['Confirmed'] * 100,2))],index=case_vs_recovered.index,columns=['Death Rate(%)'])
total_rate = pd.DataFrame([round(case_vs_recovered['Recovered'].sum() / case_vs_recovered['Confirmed'].sum() * 100, 2),round(case_vs_recovered['Death'].sum() / case_vs_recovered['Confirmed'].sum() * 100 , 2)],index=['Total Recover Rate','Total Death Rate'],columns=['Percentage(%)'])
display(recover_rate,death_rate,total_rate)
ax = case_vs_recovered.plot.bar(rot=0,figsize=(35,10),width=0.8)
plt.xlabel('Province'),plt.ylabel('Cases'),plt.autoscale()

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
ax

last_update = '4/17/20'
current_cases = confirmed
current_cases = current_cases[['Country/Region',last_update]]

current_cases = current_cases.groupby('Country/Region').sum().sort_values(by=last_update, ascending=False)

current_cases['recovered'] = recovered[['Country/Region',last_update]].groupby('Country/Region').sum().sort_values(by=last_update,ascending=False)

current_cases['deaths'] = deaths[['Country/Region',last_update]].groupby('Country/Region').sum().sort_values(by=last_update,ascending=False)

current_cases['active'] = current_cases[last_update]-current_cases['recovered']-current_cases['deaths']

current_cases = current_cases.rename(columns={last_update:'confirmed'
                                              ,'recovered':'recovered'
                                              ,'deaths':'deaths'
                                              ,'active':'active'})

current_cases.style.background_gradient(cmap='Blues')
