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
pd.set_option('display.max_rows', 12)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# -

import ipympl
# %matplotlib widget

# # data read

# matplot setting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/NGULIM.ttf").get_name()
font_name = font_manager.FontProperties(
    fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf").get_name()
rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False
# plt.style.use('dark_background')

import warnings
warnings.filterwarnings('ignore')

df_final = "../../data/가공데이터/df_final.csv"
df_final = pd.read_csv(df_final)
df_final['날짜'] = pd.to_datetime(df_final['날짜'], format='%Y-%m-%d')
df_final['week'] = df_final['날짜'].dt.week
df_final['month'] = df_final['날짜'].dt.month
df_final.head()

del df_final['Unnamed: 0']
del df_final['구_y']
df_final = df_final.drop(
    columns=[
        'pm25',
        'co2',
        'vocs',
        'noise',
        'temp',
        'humi']).dropna()

df_final.성별코드 = df_final.성별코드.replace('F', -1)
df_final.성별코드 = df_final.성별코드.replace('M', 1)


df_final.rename(columns={'구_x': '구', 'FLOW_POP_CNT': '유동인구'}, inplace=True)

map_card_code = {10: '숙박',
 20: '레저용품',
 21: '레저업소',
 22: '문화취미',
 30: '가구',
 31: '전기',
 32: '주방용구',
 33: '연료판매',
 34: '광학제품',
 35: '가전',
 40: '유통업',
 42: '의복',
 43: '작물',
 44: '신변잡화',
 50: '서적문구',
 52: '사무통신',
 60: '자동차판매',
 62: '자동차정비',
 70: '의료기관',
 71: '보건위생',
 80: '요식업소',
 81: '음료식품',
 92: '수리서비스'}
df_final['업종코드'] = pd.DataFrame(df_final['업종코드']).applymap(map_card_code.get)

# null check -> None
df_final.isna().sum()


# # data 분석

# ## 미세먼지 vs 소비(card & gs)

# ### 미세먼지 vs gs

# +
# 일정 기간내 correlation map (pm10 vs card data)
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))

state1 = df_final.month >= 1 
state2 = df_final.month <= 3 

df_pm10_gs = df_final[state1 & state2]
corr_pm10_gs = df_pm10_gs[['pm10',
                           '매출지수',
                           '식사_비중',
                           '간식_비중',
                           '마실거리_비중',
                           '홈&리빙_비중',
                           '헬스&뷰티_비중',
                           '취미&여가활동_비중',
                           '사회활동_비중',
                           '임신/육아_비중']].corr()
sns.heatmap(
    corr_pm10_gs,
    mask=np.zeros_like(
        corr_pm10_gs,
        dtype=np.bool),
    cmap=sns.diverging_palette(
            220,
            10,
            as_cmap=True),
    square=True,
    ax=ax,
    vmin=-1,
    vmax=1)

# + jupyter={"outputs_hidden": true}
# 임신육아, 헬스뷰티 0.15이상
corr_pm10_gs['pm10']


# + [markdown] toc-hr-collapsed=false
# ## 미세먼지 vs 성연령별 flow
# -

# ### year, 미세먼지 vs 성연령별 flow

# +
# 임시 func
def mise_corr(index, df_final):
    df = df_final[['날짜', '구', '동', '성별코드', '나이코드', '유동인구', index]]
    df = df.drop_duplicates(keep='first')
    df = df.drop(
        ['날짜', '구', '동'],
        axis=1).groupby(
        ['성별코드', '나이코드']).corr()['유동인구'].unstack(
        level=2).drop(
        columns='유동인구')

    return df

corr_mise_sa = mise_corr('pm10', df_final)
corr_mise_sa.style.bar(
    subset='pm10',
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)


# -

# ### week, 미세먼지 vs 성연령별 flow

def mise_week_corr(index, week, df_final):
    df = df_final[['날짜', '구', '동', '성별코드', '나이코드', '유동인구', 'pm10']]
    state1 = df_final.week  == week
    df = df[state1] 
    df = df.drop_duplicates(keep='first')
    df = df.drop( ['날짜', '구', '동'], axis=1).groupby( ['성별코드', '나이코드']).corr()['유동인구'].unstack( level=2).drop( columns='유동인구')
    df = df.rename(columns={'pm10':week})

    return df
week_all = df_final.week.unique()
week_all.sort()
corr_mise_sa_week = (mise_week_corr('pm10', week, df_final) for week in week_all)
corr_mise_sa_week_all = pd.concat(corr_mise_sa_week, axis=1)
corr_mise_sa_week_all.style.bar(
    subset=week_all,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)


# ### month, 미세먼지 vs 성연령별 flow

def mise_month_corr(index, month, df_final):
    df = df_final[['날짜', '구', '동', '성별코드', '나이코드', '유동인구', 'pm10']]
    state1 = df_final.month  == month
    df = df[state1] 
    df = df.drop_duplicates(keep='first')
    df = df.drop( ['날짜', '구', '동'], axis=1).groupby( ['성별코드', '나이코드']).corr()['유동인구'].unstack( level=2).drop( columns='유동인구')
    df = df.rename(columns={'pm10':month})

    return df
month_all = df_final.month.unique()
month_all.sort()
corr_mise_sa_month = (mise_month_corr('pm10', month, df_final) for month in month_all)
corr_mise_sa_month_all = pd.concat(corr_mise_sa_month, axis=1)
corr_mise_sa_month_all.style.bar(
    subset=month_all,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)


# ### 좋음 나쁨정도(일별)를 통한 미세먼지 vs 성연령별 flow

def mise_fcast_corr(index, fcast, df_final):
    df = df_final[['날짜', '구', '동', '성별코드', '나이코드', '유동인구', 'pm10']]
    state1 = df_final.예보  == fcast
    df = df[state1] 
    df = df.drop_duplicates(keep='first')
    df = df.drop( ['날짜', '구', '동'], axis=1).groupby( ['성별코드', '나이코드']).corr()['유동인구'].unstack( level=2).drop( columns='유동인구')
    df = df.rename(columns={'pm10':fcast})

    return df
fcast_all = df_final.예보.unique()
fcast_all.sort()
corr_mise_sa_fcast = (mise_fcast_corr('pm10', fcast, df_final) for fcast in fcast_all)
corr_mise_sa_fcast_all = pd.concat(corr_mise_sa_fcast, axis=1)
corr_mise_sa_fcast_all.style.bar(
    subset=fcast_all,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)

# ### 미세먼지 vs 성연령별 flow (기간 별)

df_final.groupby(['month']).pm10.agg({'pm10':['mean','min','max']})

df_yebo_ratio = df_final.groupby(['month']).예보.value_counts() / df_final.groupby(['month']).예보.count()

df_yebo_ratio

# 나쁨 & 매우나쁨 비율 합
df_yebo_ratio.unstack(level=0).T[['나쁨', '매우나쁨']].fillna(0).sum(axis=1)


# #### 1,2,3 월

def mise_corr(index, df_final):
    df = df_final[['날짜', '구', '동', '성별코드', '나이코드', '유동인구', index]]
    df = df.drop_duplicates(keep='first')
    df = df.drop(
        ['날짜', '구', '동'],
        axis=1).groupby(
        ['성별코드', '나이코드']).corr()['유동인구'].unstack(
        level=2).drop(
        columns='유동인구')

    return df


# +
state1 = df_final.month >= 1 
state2 = df_final.month <= 3 
# state3 = df_final.예보 == '좋음'

corr_mise_sa = mise_corr('pm10', df_final[state1 & state2])
# corr_mise_sa = mise_corr('pm10', df_final[state1 | state2 & state3])
corr_mise_sa.style.bar(
    subset='pm10',
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# #### 4~10 월

# +
state1 = df_final.month >= 4 
state2 = df_final.month <= 10 
# state3 = df_final.예보 == '좋음'

corr_mise_sa = mise_corr('pm10', df_final[state1 & state2])
# corr_mise_sa = mise_corr('pm10', df_final[state1 | state2 & state3])
corr_mise_sa.style.bar(
    subset='pm10',
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# #### 11~12 월

# +
state1 = df_final.month >= 11
state2 = df_final.month <= 12 
# state3 = df_final.예보 == '좋음'

corr_mise_sa = mise_corr('pm10', df_final[state1 & state2])
# corr_mise_sa = mise_corr('pm10', df_final[state1 | state2 & state3])
corr_mise_sa.style.bar(
    subset='pm10',
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)

# + [markdown] toc-hr-collapsed=false
# ## gs지수 vs 성연령별 유동인구 분포 확인


# + [markdown] toc-hr-collapsed=false
# ### 1~12월 gs지수 vs 성연령별 유동인구 분포 확인


# +
def sa_corr(index, df_final):
    df = df_final[['날짜', '구', '동', '성별코드', '나이코드', '유동인구', index]]
    df = df.drop_duplicates(keep='first')
    df = df.drop(
        ['날짜', '구', '동'],
        axis=1).groupby(
        ['성별코드', '나이코드']).corr()['유동인구'].unstack(
        level=2).drop(
        columns='유동인구')

    return df


# df_final_ is scaled
df_final_ = df_final.copy()
gs_index = ['식사_비중', '간식_비중', '마실거리_비중',
            '홈&리빙_비중', '헬스&뷰티_비중', '취미&여가활동_비중', '사회활동_비중', '임신/육아_비중']
for index in gs_index:
    df_final_[index] = df_final['매출지수'].mul(df_final[index])

gs_index.append('매출지수')

corr_gs_sa = (sa_corr(index, df_final_) for index in gs_index)
corr_gs_sa_all = pd.concat(corr_gs_sa, axis=1)
corr_gs_sa_all.style.bar(
    subset=gs_index,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# #### 1~3월 gs지수 vs 성연령별 유동인구 분포 확인


# +
# df_final_ is scaled
df_final_ = df_final.copy()
gs_index = ['식사_비중', '간식_비중', '마실거리_비중',
            '홈&리빙_비중', '헬스&뷰티_비중', '취미&여가활동_비중', '사회활동_비중', '임신/육아_비중']
for index in gs_index:
    df_final_[index] = df_final['매출지수'].mul(df_final[index])

gs_index.append('매출지수')

# 월별
state1 = df_final.month >= 1 
state2 = df_final.month <= 3 
corr_gs_sa = (sa_corr(index, df_final_[state1 & state2]) for index in gs_index)
corr_gs_sa_all = pd.concat(corr_gs_sa, axis=1)
corr_gs_sa_all.style.bar(
    subset=gs_index,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# #### 4~10월 gs지수 vs 성연령별 유동인구 분포 확인


# +
# df_final_ is scaled
df_final_ = df_final.copy()
gs_index = ['식사_비중', '간식_비중', '마실거리_비중',
            '홈&리빙_비중', '헬스&뷰티_비중', '취미&여가활동_비중', '사회활동_비중', '임신/육아_비중']
for index in gs_index:
    df_final_[index] = df_final['매출지수'].mul(df_final[index])

gs_index.append('매출지수')

# 월별
state1 = df_final.month >= 4 
state2 = df_final.month <= 10 
corr_gs_sa = (sa_corr(index, df_final_[state1 & state2]) for index in gs_index)
corr_gs_sa_all = pd.concat(corr_gs_sa, axis=1)
corr_gs_sa_all.style.bar(
    subset=gs_index,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# #### 11~12월 gs지수 vs 성연령별 유동인구 분포 확인


# +
# df_final_ is scaled
df_final_ = df_final.copy()
gs_index = ['식사_비중', '간식_비중', '마실거리_비중',
            '홈&리빙_비중', '헬스&뷰티_비중', '취미&여가활동_비중', '사회활동_비중', '임신/육아_비중']
for index in gs_index:
    df_final_[index] = df_final['매출지수'].mul(df_final[index])

gs_index.append('매출지수')

# 월별
state1 = df_final.month >= 11
state2 = df_final.month <= 12 
corr_gs_sa = (sa_corr(index, df_final_[state1 & state2]) for index in gs_index)
corr_gs_sa_all = pd.concat(corr_gs_sa, axis=1)
corr_gs_sa_all.style.bar(
    subset=gs_index,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)

# + [markdown] toc-hr-collapsed=false
# ## card vs 성연령별 유동인구 분포 확인
# * [성별코드, 나이코드, 업종코드]의 모든 cased에 대해서 이용금액이 존재하지 않아
# * 각 case마다 이용금액을 고려
# * (2 x 10 x 23) = 460
#     * 매우많으므로 분석방향 고려해야함
#     * 일정 corr 값을 넘는 것을 추려야함
# -


# ### 이용금액 vs flow

# corr 이용금액
def card_corr(sex, age, b_code, df_final, corr_value):
    df = df_final[['날짜', '구', '동', '성별코드', '나이코드', '업종코드','유동인구', corr_value]]
    state1 = df_final.성별코드 == sex
    state2 = df_final.나이코드 == age
    state3 = df_final.업종코드 == b_code
    df = df[state1 & state2 & state3]
    df = df.drop(
        ['날짜', '구', '동'],
        axis=1).groupby(['성별코드', '나이코드', '업종코드']).corr()['유동인구'].unstack(level=3).drop(columns='유동인구').stack().unstack(level=2)

    return df


# +
import itertools
sex_all = df_final.성별코드.unique()
age_all = df_final.나이코드.unique()
b_code_all = df_final.업종코드.unique()

corr_card_sa = (card_corr(sex, age, b_code, df_final, '이용금액') for sex, age, b_code in itertools.product(sex_all, age_all, b_code_all) )
corr_card_sa_all = pd.concat(corr_card_sa, axis=0)
corr_card_sa_all_fillna = corr_card_sa_all.sum(level=[0,1], skipna=True)
corr_card_sa_all_fillna.sort_index(inplace=True)

corr_card_sa_all_fillna.style.bar(
    subset=corr_card_sa_all_fillna.columns,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# #### 1~3월 이용금액 vs flow

# +
import itertools
sex_all = df_final.성별코드.unique()
age_all = df_final.나이코드.unique()
b_code_all = df_final.업종코드.unique()

state1 = df_final.month >= 1 
state2 = df_final.month <= 3 

corr_card_sa = (card_corr(sex, age, b_code, df_final[state1 & state2], '이용금액') for sex, age, b_code in itertools.product(sex_all, age_all, b_code_all) )
corr_card_sa_all = pd.concat(corr_card_sa, axis=0)
corr_card_sa_all_fillna = corr_card_sa_all.sum(level=[0,1], skipna=True)
corr_card_sa_all_fillna.sort_index(inplace=True)

corr_card_sa_all_fillna.style.bar(
    subset=corr_card_sa_all_fillna.columns,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# #### 4~10월 이용금액 vs flow

# +
import itertools
sex_all = df_final.성별코드.unique()
age_all = df_final.나이코드.unique()
b_code_all = df_final.업종코드.unique()

state1 = df_final.month >= 4 
state2 = df_final.month <= 10 

corr_card_sa = (card_corr(sex, age, b_code, df_final[state1 & state2], '이용금액') for sex, age, b_code in itertools.product(sex_all, age_all, b_code_all) )
corr_card_sa_all = pd.concat(corr_card_sa, axis=0)
corr_card_sa_all_fillna = corr_card_sa_all.sum(level=[0,1], skipna=True)
corr_card_sa_all_fillna.sort_index(inplace=True)

corr_card_sa_all_fillna.style.bar(
    subset=corr_card_sa_all_fillna.columns,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# #### 11~12월 이용금액 vs flow

# +
import itertools
sex_all = df_final.성별코드.unique()
age_all = df_final.나이코드.unique()
b_code_all = df_final.업종코드.unique()

state1 = df_final.month >= 11 
state2 = df_final.month <= 12 

corr_card_sa = (card_corr(sex, age, b_code, df_final[state1 & state2], '이용금액') for sex, age, b_code in itertools.product(sex_all, age_all, b_code_all) )
corr_card_sa_all = pd.concat(corr_card_sa, axis=0)
corr_card_sa_all_fillna = corr_card_sa_all.sum(level=[0,1], skipna=True)
corr_card_sa_all_fillna.sort_index(inplace=True)

corr_card_sa_all_fillna.style.bar(
    subset=corr_card_sa_all_fillna.columns,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# ### 이용건수(sum)

# 이용건수 확인
df_card_num = df_final[['성별코드', '나이코드', '업종코드', '이용건수']]
df_card_num = df_card_num.groupby(['성별코드', '나이코드', '업종코드']).sum().unstack(level=2)
df_card_num.style.bar(
    subset=df_card_num.columns,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=df_card_num.min().min(),
    vmax=df_card_num.max().max())

# ### 이용건수(sum) vs flow

# +
import itertools
sex_all = df_final.성별코드.unique()
age_all = df_final.나이코드.unique()
b_code_all = df_final.업종코드.unique()

corr_card_sa_num = (card_corr(sex, age, b_code, df_final, '이용건수') for sex, age, b_code in itertools.product(sex_all, age_all, b_code_all) )
corr_card_sa_num_all = pd.concat(corr_card_sa_num, axis=0)
corr_card_sa_num_all_fillna = corr_card_sa_num_all.sum(level=[0,1], skipna=True)
corr_card_sa_num_all_fillna.sort_index(inplace=True)

corr_card_sa_num_all_fillna.style.bar(
    subset=corr_card_sa_num_all_fillna.columns,
    align='mid',
    color=[
        '#d65f5f',
        '#5fba7d'],
    vmin=-1,
    vmax=1)
# -

# # profiling

import pandas_profiling
pandas_profiling.ProfileReport(
                               df_final
                               [['pm10', '성별코드', '나이코드', '이용건수', '이용금액',
                                 'FLOW_POP_CNT']].sample(n=10000))

# # TEST CODE

df_final[['예보','pm10']].groupby('예보').agg(['min', 'max'])

# 월별 미세먼지 나쁨이상 빈도수
df_final.groupby(['날짜']).pm10.agg(['mean', 'min', 'max'])

df_final.groupby(['날짜']).pm10.describe()

df_std_ratio = df_final.groupby(['날짜']).pm10.describe()['std'] / df_final.groupby(['날짜']).pm10.describe()['mean']

df_std_ratio[df_std_ratio > 0.5]


