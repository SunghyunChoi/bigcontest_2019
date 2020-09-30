# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# +
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %matplotlib inline
# -

card = '../../data/가공데이터/카드매출.csv'
gs = '../../data/가공데이터/유통데이터.csv'
flow_time = '../../data/가공데이터/유동인구_시간.csv'
flow_sexage = '../../data/가공데이터/유동인구_성연령.csv'
data_mi = '../../data/가공데이터/환경기상데이터.csv'

df_card = pd.read_csv(card)
df_gs = pd.read_csv(gs)
df_flow_t = pd.read_csv(flow_time)
df_flow_sa = pd.read_csv(flow_sexage)
df_mi = pd.read_csv(data_mi, encoding = 'euc-kr')

df_card['나이코드'] = df_card['나이코드'].astype(str)
df_card.info()
df_card.head()

df_gs.info()
df_gs.head()

df_flow_t.info()
df_flow_t.head()

df_flow_sa.info()
df_flow_sa.head()

df_mi.info()
df_mi = df_mi.rename(columns = {'행정동' : '동'})
df_mi.head()

# ## 유동인구 데이터 변환(카드 데이터와 합치기 위해)

# +
#0~25살, 65세 이상 데이터 하나로 합침

df_flow_sa['MAN_FLOW_POP_CNT_0024'] = df_flow_sa['MAN_FLOW_POP_CNT_0004'] + df_flow_sa['MAN_FLOW_POP_CNT_0509'] + df_flow_sa['MAN_FLOW_POP_CNT_1014'] + df_flow_sa['MAN_FLOW_POP_CNT_1519'] + df_flow_sa['MAN_FLOW_POP_CNT_2024']
df_flow_sa['WMAN_FLOW_POP_CNT_0024'] = df_flow_sa['WMAN_FLOW_POP_CNT_0004'] + df_flow_sa['WMAN_FLOW_POP_CNT_0509'] + df_flow_sa['WMAN_FLOW_POP_CNT_1014'] + df_flow_sa['WMAN_FLOW_POP_CNT_1519'] + df_flow_sa['WMAN_FLOW_POP_CNT_2024']
df_flow_sa['MAN_FLOW_POP_CNT_65U'] = df_flow_sa['MAN_FLOW_POP_CNT_6569'] + df_flow_sa['MAN_FLOW_POP_CNT_70U']
df_flow_sa['WMAN_FLOW_POP_CNT_65U'] = df_flow_sa['WMAN_FLOW_POP_CNT_6569'] + df_flow_sa['WMAN_FLOW_POP_CNT_70U']

df_flow_sa = df_flow_sa.drop(columns = ['MAN_FLOW_POP_CNT_0004', 'MAN_FLOW_POP_CNT_0509', 'MAN_FLOW_POP_CNT_1014', 'MAN_FLOW_POP_CNT_1519', 'MAN_FLOW_POP_CNT_2024','MAN_FLOW_POP_CNT_6569', 'MAN_FLOW_POP_CNT_70U', 'WMAN_FLOW_POP_CNT_0004', 'WMAN_FLOW_POP_CNT_0509', 'WMAN_FLOW_POP_CNT_1014', 'WMAN_FLOW_POP_CNT_1519', 'WMAN_FLOW_POP_CNT_2024', 'WMAN_FLOW_POP_CNT_6569', 'WMAN_FLOW_POP_CNT_70U'])
df_flow_sa.head()
                                        

# +
columns_flow = df_flow_sa.columns.tolist()
columns_man = [d for d in columns_flow if 'MAN' in d and 'WMAN' not in d]
columns_woman = [d for d in columns_flow if 'WMAN' in d]
columns_default = ['날짜', '동']

df_flow_man = df_flow_sa[columns_default + columns_man]
df_flow_woman = df_flow_sa[columns_default + columns_woman]

df_flow_man.insert(2, '성별코드', 'M')
df_flow_woman.insert(2, '성별코드', 'F')

columns_man_new = [d[4:] for d in df_flow_man.columns.tolist() if 'MAN' in d]
columns_woman_new = [d[5:] for d in df_flow_woman.columns.tolist() if 'WMAN' in d]
columns_default = ['날짜', '동', '성별코드']

df_flow_man.columns = columns_default + columns_man_new
df_flow_woman.columns = columns_default + columns_woman_new

df_flow_sex = pd.concat([df_flow_man,df_flow_woman], sort = False)

# +
columns_flow_age = df_flow_sex.columns.tolist()
df_flow_age = []

for i in range(10):
    
    columns_age = columns_flow_age[3:]
    df_flow_age.append(df_flow_sex[columns_default + [columns_age[i]]])
    df_flow_age[i].insert(3, '나이코드', columns_age[i][-4:-2])
    df_flow_age[i] = df_flow_age[i].rename(columns = {df_flow_age[i].columns.tolist()[-1] : df_flow_age[i].columns.tolist()[-1][0:12]})
    
df_flow_final = pd.concat(df_flow_age, sort = False)



df_flow_final['나이코드'] = df_flow_final['나이코드'].replace('_6', '65')
df_flow_final['나이코드'] = df_flow_final['나이코드'].replace('00', '20')

df_flow_final.info()
df_flow_final.head()

# +
df_card_flow = pd.merge(df_card, df_flow_final ,on = ['날짜', '동', '성별코드', '나이코드'])

df_card_flow.info()
df_card_flow.head(100)
# -

# ## Dataframe 합치기

# 결제 데이터 : 일 - 구 - 행정동 + 성별코드 + 나이코드
#
# 나머지 데이터 : 일 - 구 - 행정동

# +
df_mi['날짜'] = df_mi['날짜'].astype(str)

df_mi['날짜'].head()

# -

for df in [df_card, df_gs, df_flow_t, df_flow_sa, df_mi, df_card_flow]:
    df['날짜'] = pd.to_datetime(df['날짜'], format = '%Y-%m-%d')

a = pd.merge(df_mi, df_gs ,on = ['날짜', '구', '동'])
a = pd.merge(a, df_card_flow, on = ['날짜','구', '동'])
a.head(10000)

a.to_csv('df_final.csv')

# +
data_mi2 = df_mi.dropna(subset = ['pm10'])
print(len(df_mi) - np.sum(df_mi.pm10.isnull()))

data_mi_pivot = data_mi2.pivot_table('pm10', index = ['날짜', '구', '행정동'], columns = ['시'])


for i in range(53):
    data_mi_pivot.iloc[i].plot()
# -

# # 요일별 이용금액 파악

# +


data_mi_day = (df_mi.groupby(by = '날짜', as_index = False).mean())[['pm10', '날짜']]
df_card_day = (df_card.groupby(by = '날짜', as_index = False).mean())
for df in [df_card_day, data_mi_day]:
    df['날짜'] = pd.to_datetime(df['날짜'], format = '%Y-%m-%d')

    
df_card_day['week'] = df_card_day['날짜'].dt.weekday
data_mi_day['week'] = df_card_day['날짜'].dt.weekday
    
a = np.linspace(0, 365, 12)
for i in range(len(a)-1):
    #subplot(3, 4, )
    df_test = df_card_day.iloc[int(a[i]):int(a[i+1])]
    print(df_test.head())
    plt.bar(x=df_card_day['week'], height = df_card_day['이용금액'])
    plt.show()
    
#plt.bar(x=df_card_day['week'], height = df_card_day['이용금액'])

#plt.show()
# -

# # 성별 소비량 비교

# +
df_card_sex = df_card

#print(df_card_sex_m.head())
df_card_sex_m = df_card_sex[df_card_sex['성별코드']=='M'].groupby(by = '날짜', as_index = False).mean()
df_card_sex_w = df_card_sex[df_card_sex['성별코드']=='F'].groupby(by = '날짜', as_index = False).mean()

for df in [df_card_sex_m, df_card_sex_w]:
    df['날짜'] = pd.to_datetime(df['날짜'], format = '%Y-%m-%d')

df_card_sex_m['week'] = df_card_sex_m['날짜'].dt.weekday
df_card_sex_w['week'] = df_card_sex_w['날짜'].dt.weekday

df_card_sex_m.head()
# -

p1 = plt.bar(df_card_sex_w['week'], height = df_card_sex_w['이용금액'])
p2 = plt.bar(df_card_sex_m['week'], height = df_card_sex_m['이용금액'], bottom = df_card_sex_w['이용금액'])
plt.legend((p1[0], p2[0]), ('Women', 'Men'), loc = 'upper right')

# # 연령대별 소비 비교

# +
# df.pivot_table(values='val', index=df.index, columns='key', aggfunc='first')

df_card_a = df_card[['날짜', '나이코드', '이용금액']]

for df in [df_card_a]:
    df['날짜'] = pd.to_datetime(df['날짜'], format = '%Y-%m-%d')

df_card_a['week'] = df_card_a['날짜'].dt.weekday
df_card_age = df_card_a.pivot_table(values = '이용금액', index = 'week', columns = '나이코드' )
df_card_age.plot(kind = 'bar', stacked = True)

