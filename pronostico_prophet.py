#%%
#3ra parte input estacionales prophet
#####################################

import sys
from xml.dom.expatbuilder import CDATA_SECTION_NODE
from fx_app import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

#from fx_app import rename_variables
pd.set_option('float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None) 

path = "~"
pathd = "~"



hora_empieza = datetime.today()
print("Empieza script prophet", hora_empieza)

df = pd.read_csv(pathd+"/base_suc.csv") 

########################### PROPHET ###############################
###### PARÁMETROS ######
# df base - preferible traer los datos crudos de las querys. Pero primero es necesario crear csv feriados acorde a como lo requiere el prophet. 


# parámetros prophet  --> se usa modelo WM_best 2021
#t='1'
#param_grid = {'growth':'linear','n_changepoints':30, 'changepoint_prior_scale':0.02,
#              'changepoint_range':0.85, 'seasonality_mode':'additive',
#             'seasonality_prior_scale':6.0,'yearly_seasonality':2,
#              'weekly_seasonality':4,'daily_seasonality':4,'holidays_prior_scale':10}

#t='2'
#param_grid = {'growth':'linear','n_changepoints':25,'changepoint_prior_scale':0.05,
#                'changepoint_range':0.8,'seasonality_mode':'additive',
#                'seasonality_prior_scale':10.0,'yearly_seasonality':'auto',
#               'weekly_seasonality':'auto','daily_seasonality':False,'holidays_prior_scale':10}

#t='3'
#param_grid = {'growth':'linear','n_changepoints':30, 'changepoint_prior_scale':0.001,
#              'changepoint_range':0.9, 'seasonality_mode':'additive',
#             'seasonality_prior_scale':10.0,'yearly_seasonality':2,
#              'weekly_seasonality':30,'daily_seasonality':30}


t='4'
param_grid = {'growth':'linear','n_changepoints':30, 'changepoint_prior_scale':0.1,
              'changepoint_range':0.9, 'seasonality_mode':'additive',
             'seasonality_prior_scale':5,'yearly_seasonality':2,
             'weekly_seasonality':6,'daily_seasonality':12}

day_desde=365

# parámetros fechas test-pronóstico
#desde_test = (pd.to_datetime(df.fecha.max())- pd.DateOffset(days=6)).strftime("%Y-%m-%d") 
#hasta_test = (pd.to_datetime(df.fecha.max())).strftime("%Y-%m-%d") 
#fin2=(pd.to_datetime(desde_test) - pd.DateOffset(days=day_desde)).strftime("%Y-%m-%d") 
desde_test = '2022-05-26'
hasta_test = '2022-06-01'
fin2='2021-06-02'




df=df.merge(zon_cen,on='locales',how='inner')
print(df.shape)
#df=df[df['articulo'].isin(ar)]
df['fecha'] = pd.to_datetime(df['fecha'])
df["articulo"] = df["articulo"].astype(int)
df["locales"] = df["locales"].astype(int)

d1 = pd.DataFrame({'fecha': pd.date_range(pd.to_datetime(desde_test), pd.to_datetime(hasta_test))})
d2 = df[['articulo','locales', 'zona']].drop_duplicates()
d = d1.assign(dummy=1).merge(d2.assign(dummy=1), on='dummy', how='outer').drop('dummy', axis=1)
df = pd.concat([df, d], ignore_index=True)
df['venta_kg'] = df['venta_kg'].fillna(0)



print('prophet')
dfouts = calcular_prophet(df, param_grid=param_grid,nivel='zona',desde=desde_test,hasta=hasta_test,fin=fin2)

print()
print('termino prophet')

dfouts['ds']= pd.to_datetime(dfouts['ds']).dt.strftime('%Y-%m-%d') 
dfouts = dfouts.rename(columns={"ds":"fecha"})
dfouts['yhat_lower']=np.where(dfouts['yhat_lower']<0,0,dfouts['yhat_lower'])
dfouts['yhat']=np.where(dfouts['yhat']<0,0,dfouts['yhat'])  #modificar por arreglos 




r=1
dfouts.to_csv(pathd+f"/pronostico_prophet_{t}_zona_futuro_{r}.csv", index=False)
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print("****************************************************************** ")
print("Termina prophet")
print("****************************************************************** ")
print("Periodo desde: ", fin2, " hasta ", hasta_test)
print("****************************************************************** ")
print("tiempo de computo: Empezó: ", hora_empieza, " Terminó", datetime.today())
print("****************************************************************** ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")

