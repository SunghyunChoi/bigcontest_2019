# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + colab={} colab_type="code" id="5qO0LoVFyoCG"
import os
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt
# -

import ipympl
# %matplotlib inline

# # data read

# matplot setting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/gulim.ttc").get_name()
#font_name = font_manager.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()
rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False
# plt.style.use('dark_background')

import warnings 
warnings.filterwarnings('ignore')

# +
df_card_flow = "../../data/가공데이터/df_final.csv"

df_card_flow = pd.read_csv(df_card_flow)
# -

df_card_flow.head()

#del df_card_flow['Unnamed: 0']
df_card_flow = df_card_flow.drop(columns = ['co2', 'vocs', 'noise', 'temp', 'humi', 'pm25'])

df_card_flow = df_card_flow.dropna()

df_card_flow.info()

df_card_flow.성별코드 = df_card_flow.성별코드.replace('F', -1)
df_card_flow.성별코드 = df_card_flow.성별코드.replace('M', 1)

# + jupyter={"outputs_hidden": true}
import pandas_profiling
pandas_profiling.ProfileReport(df_card_flow[['pm10', '성별코드', '나이코드', '이용건수', '이용금액', 'FLOW_POP_CNT']].sample(n=1000), check_correlation = True)
# -

df_card_flow = df_card_flow.drop(columns = ['구_y', 'Unnamed: 0'])
df_card_flow = df_card_flow.rename(columns = {'구_x' : '구'})

# +
for df in [df_card_flow]:
    df['날짜'] = pd.to_datetime(df['날짜'], format = '%Y-%m-%d')

df_card_flow['week'] = df_card_flow['날짜'].dt.weekday

# +
age_list = set(df_card_flow['나이코드'].values)
sex_list = set(df_card_flow['성별코드'].values)
week_list = set(df_card_flow['week'].values)

sex_age_data = []
sex_age_data_name = []

for age,sex,day in itertools.product(age_list, sex_list, week_list):
    data_name = 'data_' + str(age) + '_' + str(sex) + str(day)
    #print(df_card_flow[(df_card_flow['나이코드'] == age) & (df_card_flow['성별코드'] == sex)].head())
    sex_age_data.append(df_card_flow[(df_card_flow['나이코드'] == age) & (df_card_flow['성별코드'] == sex) & (df_card_flow['week'] == day)])
    sex_age_data_name.append(data_name)
    
# -

df_card_flow.info()

# + jupyter={"outputs_hidden": true}
for name,data in zip(sex_age_data_name, sex_age_data):
    print(name)
    data = data[['pm10', '이용금액', 'FLOW_POP_CNT']]
    print(data.corr())
    #corr_data = data.corr()
    #plt.plot()
    #plt.imshow(corr_data,cmap='hot',interpolation='nearest')
    #plt.show()
# -

df_card_flow.head()

# +
import os

os.getcwd()
