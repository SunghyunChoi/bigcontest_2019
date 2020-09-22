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
import seaborn as sns
import random
# -

import ipympl
# %matplotlib inline
# %matplotlib widget

# # data read

# matplot setting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
fname="c:/Windows/Fonts/batang.ttc"
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/batang.ttc").get_name()
#font_name = font_manager.FontProperties(fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()
rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False
# plt.style.use('dark_background')

# +
import matplotlib
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
# font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
font_location = 'C:/Windows/Fonts/NanumGothic.ttf' # For Windows
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

# %matplotlib inline

# -

import warnings 
warnings.filterwarnings('ignore')

df_card_flow_dir = "../../data/가공데이터/df_final.csv"
df_shop_list = "업종코드.csv"

df_card_flow = pd.read_csv(df_card_flow_dir)

df_shop = pd.read_csv(df_shop_list, encoding = 'euc-kr')

#del df_card_flow['Unnamed: 0']
df_card_flow = df_card_flow.drop(columns = ['co2', 'vocs', 'noise', 'temp', 'humi', 'pm25'])

df_card_flow = df_card_flow.dropna()

df_card_flow.성별코드 = df_card_flow.성별코드.replace('F', -1)
df_card_flow.성별코드 = df_card_flow.성별코드.replace('M', 1)
df_card_flow.예보 = df_card_flow.예보.replace('좋음', 1)
df_card_flow.예보 = df_card_flow.예보.replace('보통', 1)
df_card_flow.예보 = df_card_flow.예보.replace('나쁨', -1)
df_card_flow.예보 = df_card_flow.예보.replace('매우나쁨', -1)


df_card_flow = df_card_flow.drop(columns = ['구_y', 'Unnamed: 0'])
df_card_flow = df_card_flow.rename(columns = {'구_x' : '구'})

# +
for df in [df_card_flow]:
    df['날짜'] = pd.to_datetime(df['날짜'], format = '%Y-%m-%d')

df_card_flow['week'] = df_card_flow['날짜'].dt.weekday

for df in [df_card_flow]:
    df['날짜'] = pd.to_datetime(df['날짜'], format = '%Y-%m-%d')

df_card_flow['month'] = df_card_flow['날짜'].dt.month

# -

df_card_flow.head()

a = (df_card_flow.groupby('month')['pm10'].describe())[['mean', 'std', 'min', 'max']].T
a[[4, 5, 6, 7, 8, 9, 10, 11, 12]].merge(a[[1,2, 3]], left_index = True, right_index = True)


# +
#df_card_flow[df_card_flow['성별코드'] == -1].groupby('날짜').mean()['FLOW_POP_CNT'].plot()
#df_card_flow[df_card_flow['성별코드'] == 1].groupby('날짜').mean()['FLOW_POP_CNT'].plot()

df_card_flow.groupby('날짜').sum()['FLOW_POP_CNT'].plot(figsize = (15,4))
# -

a = (df_card_flow.groupby('month')['FLOW_POP_CNT'].describe())[['mean', 'std', 'min', 'max']].T
a[[4, 5, 6, 7, 8, 9, 10, 11, 12]].merge(a[[1,2, 3]], left_index = True, right_index = True)


df_card_flow.groupby('날짜').sum()['이용금액'].plot(figsize = (15,4))

# +
test = df_card_flow.groupby('날짜', as_index = False).sum()

test['month'] = test['날짜'].dt.month

a = (test.groupby('month')['이용금액'].describe())[['mean', 'std', 'min', 'max']].T
a[[4, 5, 6, 7, 8, 9, 10, 11, 12]].merge(a[[1,2, 3]], left_index = True, right_index = True)

# +
#df_card_flow[df_card_flow['성별코드'] == -1].groupby('날짜').mean()['FLOW_POP_CNT'].plot()
#df_card_flow[df_card_flow['성별코드'] == 1].groupby('날짜').mean()['FLOW_POP_CNT'].plot()

df_card_flow.groupby('날짜').mean()['매출지수'].plot(figsize = (15,4))

# +


a = df_card_flow.groupby('month')['매출지수'].describe()[['mean', 'std', 'min', 'max']].T
a[[4, 5, 6, 7, 8, 9, 10, 11, 12]].merge(a[[1,2, 3]], left_index = True, right_index = True)
# -

# # 월별 예보 지수 확인

# +
df_card_flow_fc = df_card_flow

df_card_flow_fc = df_card_flow_fc.groupby(by = '날짜', as_index = False).mean()
dt = df_card_flow_fc['pm10']
df_card_flow_fc['예보'] = np.where(dt > 150, '-1', 
                             np.where(dt > 80, '-1',
                                     np.where(dt > 30, '1', '1')))
df_card_flow_fc['예보'] = df_card_flow_fc['예보'].astype(int)
#df_card_flow_fc_n = df_card_flow_fc[(df_card_flow_fc['month'] == 1) | (df_card_flow_fc['month'] == 2) | (df_card_flow_fc['month'] == 3)] 
df_card_flow_fc_month = df_card_flow_fc.groupby(by = 'month').mean()
df_card_flow_fc.groupby('month')['pm10','예보'].describe()

# -

# # 월별 미세먼지 분포 확인(얼마나 정규분포에 가까운지)
#
#  1. 월 리스트 뽑기
#  2. 리스트별로 pm10 데이터의 hist 그리기
#  --> 31개밖에 안되는 데이터라 정규분포를 논하는 의미가 없음.
#  3. 1월 중심으로 먼저 비교해보기
#  
#  매출 지수 / 이용금액 / 유동인구는 각각 비교해볼 필요가 있음.
#  매출 지수 같은경우는 평균을 내서 한번에 비교 가능

# +
#2. 리스트별로 pm10 데이터의 hist 그리기

month_list = range(1,13)

for mon in month_list:
    
    print('month : ', mon)
    df_mon = df_card_flow_fc[df_card_flow_fc['month'] == mon]
    plt.hist(df_mon.pm10, bins = 100)
    #plt.show()

# +
#1월달 내의 좋음 데이터 안좋음 gs매출 데이터 비교, 같은 요일 기준 비교, 큰 유의차 없음
#1월달 데이터

df_1 = df_card_flow_fc[df_card_flow_fc['month'] == 1]

#1월달의 안좋음 데이터

df_1_b = df_1[df_1['예보'] == -1]

#1월달의 좋음 데이터

df_1_g = df_1[df_1['예보'] == 1]

g_1 = df_1_g.describe()
b_1 = df_1_b.describe()

#print(df_1_g.iloc[4] - df_1_b.iloc[3])
#print(df_1_g.iloc[5] - df_1_b.iloc[4])
#print(df_1_g.iloc[6] - df_1_b.iloc[5])
#print(df_1_g.iloc[9] - df_1_b.iloc[8])
#print(df_1_g.iloc[12] - df_1_b.iloc[9])
#print(df_1_g.iloc[13] - df_1_b.iloc[10])
#print(df_1_g.iloc[14] - df_1_b.iloc[11])

a1 = (df_1_g.iloc[4] - df_1_b.iloc[3])
a2 = (df_1_g.iloc[5] - df_1_b.iloc[4])
a3 = (df_1_g.iloc[6] - df_1_b.iloc[5])
a4 = (df_1_g.iloc[9] - df_1_b.iloc[8])
a5 = (df_1_g.iloc[12] - df_1_b.iloc[9])
a6 = (df_1_g.iloc[13] - df_1_b.iloc[10])
a7 = (df_1_g.iloc[14] - df_1_b.iloc[11])

print(a1 + a2 + a3 + a4 + a5 + a6 + a7)

#print(df_1_g[['날짜', 'week']])
#print(df_1_b[['날짜', 'week']])
# -

# # 요일 별 이용금액
#
# 1. 요일별로 미세먼지 vs 이용금액 품목 corr를 비교한다.
# 2. 11~3월 데이터 소팅
# 3. 소팅한 데이터 내의 pivot table 확인

# +
df_card_flow_shop = df_card_flow.merge(df_shop, on='업종코드')

df_card_flow_shop.head()

# -

# # 18.11월 ~ 19.3월 까지의 좋음(보통, 좋음) vs 나쁨(나쁨, 매우나쁨) 신한카드 데이터 비교

# +
month_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
df_target = df_card_flow_shop[df_card_flow_shop.month.isin(month_list)]

df_target_list = df_card_flow_fc[df_card_flow_fc.month.isin(month_list)]

df_target_g = df_target[df_target['예보'] == 1]
df_target_b = df_target[df_target['예보'] == -1]

df_target_g_list = df_target_list[df_target_list['예보'] == 1]
df_target_b_list = df_target_list[df_target_list['예보'] == -1]


print(df_target_g.shape,df_target_b.shape)
# -

a = df_target_b.groupby('날짜').mean()
b = a.groupby('month').sum()
b

# +
day_g = list(set(df_target_g_list.날짜.values))
day_b = list(set(df_target_b_list.날짜.values))
sample_size = len(day_b)
day_g_115 = random.sample(day_g, k = sample_size)
day_b_115 = random.sample(day_b, k = sample_size)

print(len(day_g), len(day_b))
print(sample_size)

df_target_g_rand = df_target_g[df_target_g['날짜'].isin(day_g_115)]
df_target_b_rand = df_target_b[df_target_b['날짜'].isin(day_b_115)]

print(len(set(day_g_115)))
print(len(list(set(df_target_g_rand.날짜.values))))
print(len(list(set(df_target_b_rand.날짜.values))))
df_target_g_rand['예보'] = 1
df_target_b_rand['예보'] = -1

df_target = pd.concat([df_target_g_rand, df_target_b_rand])

# -

df_target.corr()

# +
shop_list = set(df_target['업종명'].values)
sex_list = set(df_target['성별코드'].values)
age_list = set(df_target['나이코드'].values)

#print(shop_list)

shop_data = []
shop_data_name = []

for age,sex,shop in itertools.product(age_list, sex_list, shop_list):
    #print(shop)
    data_name = 'data_' + str(age) + '_' + str(sex) + str(shop)
    shop_data.append(df_target[(df_target['나이코드'] == age) & (df_target['성별코드'] == sex) & (df_target['업종명'] == shop)])
    shop_data_name.append(data_name)

# -

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# +
shop_data_new = []

for data in shop_data:
    data_new = remove_outlier(data, '이용금액')
    shop_data_new.append(data_new)

# + jupyter={"outputs_hidden": true}
item_list = []

for name,data in zip(shop_data_name, shop_data):
    data3 = data
    #print(data.columns.tolist())
    
    
    corr = data3[['예보', '이용금액']].corr()
    
    
    if(((np.sum(data3['이용건수'])) > 10000) & ((np.sum(np.sum(corr)) > 2.2) | (np.sum(np.sum(corr)) < 1.8))):
        print((np.sum(data3['이용건수'])))
        print((len(data3['이용건수'])))
        print(np.sum(np.sum(corr)))
        print(name)
        item_list.append(name + str(np.sum(np.sum(corr))))
        print('\n')
        print(corr)
        
        data3_g = data3[data3.날짜.isin(day_g_115)]
        data3_b = data3[data3.날짜.isin(day_b_115)]
        fig = plt.figure(1, figsize=(9, 6))
        plt.subplot(1,2,1)
        
        data3_gb = [data3_g['이용금액'], data3_b['이용금액']]
        plt.xlabel('Good       Bad')
        plt.boxplot(data3_gb)
        q4 = data3['이용금액'].quantile(1)
        plt.ylim(0,q4)
        
        plt.subplot(1,2,2)
        plt.hist(data3_g.이용금액, label = 'good', bins = 100, alpha = 0.5)
        plt.hist(data3_b.이용금액, label = 'bad', bins = 100, alpha = 0.5)
        plt.legend(loc = 'upper right')
        plt.show()


# + jupyter={"outputs_hidden": true}
sex_list = set(df_target['성별코드'].values)
age_list = set(df_target['나이코드'].values)

sex_age_data = []
sex_age_data_name = []

for age,sex in itertools.product(age_list, sex_list):
    
    data_name = 'data_' + str(age) + '_' + str(sex)
    sex_age_data.append(df_target[(df_target['나이코드'] == age) & (df_target['성별코드'] == sex)])
    sex_age_data_name.append(data_name)

# + jupyter={"outputs_hidden": true}
sex_age_item_list = []

for name,data in zip(sex_age_data_name, sex_age_data):
    data3 = data
    #print(data.columns.tolist())
    
    
    corr = data3[['예보', 'FLOW_POP_CNT']].corr()
    
    
    if(True):
        print(np.sum(np.sum(corr)))
        print(name)
        item_list.append(name + str(np.sum(np.sum(corr))))
        print('\n')
        print(corr)
        
        data3_g = data3[data3.날짜.isin(day_g_115)]
        data3_b = data3[data3.날짜.isin(day_b_115)]
        fig = plt.figure(1, figsize=(9, 6))
        plt.subplot(1,2,1)
        
        data3_gb = [data3_g['FLOW_POP_CNT'], data3_b['FLOW_POP_CNT']]
        plt.xlabel('Good       Bad')
        plt.boxplot(data3_gb)
        q4 = data3['FLOW_POP_CNT'].quantile(1)
        plt.ylim(0,q4)
        
        plt.subplot(1,2,2)
        plt.hist(data3_g.FLOW_POP_CNT, label = 'good', bins = 100, alpha = 0.5)
        plt.hist(data3_b.FLOW_POP_CNT, label = 'bad', bins = 100, alpha = 0.5)
        plt.legend(loc = 'upper right')
        plt.show()
        #유동인구는 전 연령대에서 감소한다.
# -

item_list

# + jupyter={"outputs_hidden": true}
for name,data in zip(shop_data_name, shop_data):
    data3 = data
    corr = data3[['예보', '이용금액']].corr()
    
    if(((np.sum(data3['이용건수'])) > 10000)& ((np.sum(np.sum(corr)) > 2.2) | (np.sum(np.sum(corr)) < 1.8))):
        print((np.sum(data3['이용건수'])))
        print((len(data3['이용건수'])))
        print(np.sum(np.sum(corr)))
        print(name)
        print('\n')
        print(corr)
       


# +
#업종코드별 이용금액 분포 확인하기

shop_time_data = []
shop_time_data_name = []

for shop in shop_list:
    data_name = 'data_' + str(shop)
    shop_time_data.append(df_card_flow_shop[(df_card_flow_shop['업종명'] == shop)])
    shop_time_data_name.append(data_name)

# + jupyter={"outputs_hidden": true}
for i in range(30):
    shop_plt_data_i = shop_time_data[i].groupby(by='날짜',as_index=False).sum()
    ##x = plt.plot(shop_plt_data['날짜'], shop_plt_data['이용금액'])
    plt.hist(shop_plt_data_i['이용금액'], bins = 100, normed = True)
    print(shop_time_data_name[i])
    print(np.mean(shop_plt_data_i['이용금액']))
    print(np.std(shop_plt_data_i['이용금액']))
    plt.show()


