#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
#from funciones_testeo import *
from predictores import *

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)



hora_empieza = datetime.today()
print(" Empezó Regresion: ", hora_empieza)



pathd = "--"


def corregir_pronostico(data):
    data["fecha"] = pd.to_datetime(data['fecha'], format='%Y-%m-%d')
    data['pronostico'] = np.where(data['pronostico'] <= 0, data['venta_minima_kg'], data['pronostico'])
    #data['pronostico'] = np.where((data["pronostico"] > (data["venta_maxima_kg"] * 3)), data["venta_maxima_kg"], data["pronostico"]) #revisar outlier
    return data

#zonas de precio segun sucurslaes centro




df=pd.read_csv(pathd+"/base_suc.csv")
df.pct_desc=df.pct_desc/100 #(como porcentaje entre 0 y 1)
#cuando se accione
#hasta = (pd.to_datetime(df.fecha.max())).strftime("%Y-%m-%d")   
#desde=(pd.to_datetime(hasta_test) - timedelta(days=365)).strftime("%Y-%m-%d") 

#df=df[(df.fecha>=desde)&(df.fecha<=hasta)]

df_reg=df.merge(zon_cen,on='locales',how='inner').drop('locales', 1)

df_reg=df_reg[(df_reg.fecha>='2021-06-02')&(df_reg.fecha<='2022-06-02')]

#########agrego la salida del prophet 
t='4'

r=1
dfout=pd.read_csv(pathd+f"/pronostico_prophet_{t}_zona_futuro_{r}.csv")
dfout['fecha']= pd.to_datetime(dfout['fecha']).dt.strftime('%Y-%m-%d')

# merge con la base calculada por fecha-art-zona
df_reg=df_reg.merge(dfout,on=['fecha','articulo', "zona"],how='inner')
df_reg=df_reg.replace([np.inf, -np.inf], 0).fillna(0)

#################################
""" no se acciona
df_reg=df_reg.groupby(['fecha','articulo','zona']).agg({"venta_kg":'sum',

                'yhat':'sum',
                'yhat_upper':'max',
                'yhat_lower':'min'
                
                }).reset_index() 
                

#  'kg_dia_semana_anterior','promo_volumen', 'pm_precio_r49','pm_precio_r35', 'dpm_precio_r35', 
# 'dpm_precio_r49','tend_dia_kg',  'tend_median_semanal_kg', 'tend_semanal_kg'] 
 #,'tend_dia_kg',  'tend_median_semanal_kg', 'tend_semanal_kg'
#'yhat', 'yhat_upper', 'yhat_lower' ,
#'kg_dia_semana_anterior', 
 # 'feriado',  #'cerrado', #'feriado_ant', #'feriado_pos', 'anio', #'mes_anio', 'semana',# 'dia_semana', 'grupo',
 #'tend_median_semanal_kg', #'tend_semanal_kg', #'kg_dia_semana_anterior',
"""
#####################################################v
print("Base preparada para Modelar. Duración: ", (datetime.today()-hora_empieza))

#%%Modelo Regresion Multiple

# Fechas testeo:
fecha_desde ='2022-05-26'  # df_reg.fecha.sort_values(ascending=True).unique()[-6]
fecha_hasta ='2022-06-01' # df_reg.fecha.max()

#coef = pd.DataFrame()
test_regrmult = pd.DataFrame()
niveles_zon=df_reg.zona.unique().tolist()
nivel='zona'


#defino variable para la regresion multiple
h='29'
#cols1=[ 'yhat','pct_desc', 'promo_volumen', 'dpm_precio_r42', 'pm_precio_r49',  'precio_r','pm_precio_r35', 'dpm_precio_r35',  'pm_precio_r42',  'dpm_precio_r49']
#'precio_avg_art','precio_avg_grupo' , 
cols2=['yhat', 'pct_desc','kg_dia_semana_anterior', 'promo_volumen', 
       'precio_r', 'pm_precio_r35', 'dpm_precio_r35','pm_precio_r42',
       'dpm_precio_r42', 'dpm_precio_r49', 'pm_precio_r49'] 
#cols3=['pct_desc','kg_dia_semana_anterior','tend_dia_kg','tend_median_semanal_kg','tend_semanal_kg', 'promo_volumen',   'dpm_precio_r42', 'pm_precio_r49', 'precio_r', 'pm_precio_r35', 'dpm_precio_r35',      'pm_precio_r42',  'dpm_precio_r49']
#cols4=['precio_r','tend_dia_kg','pct_desc','dpm_precio_r42',  'pm_precio_r42']
#cols5=['yhat','precio_r','tend_dia_kg','pct_desc','pm_precio_r35', 'dpm_precio_r35', 'dpm_precio_r42', 'pm_precio_r42', 'pm_precio_r49',  'dpm_precio_r49','pm_precio_r35']

cols=cols2




#seleccionar para cada zona tomar los 30 articulos de mas ventas. considerar 1 año de train
t=df_reg.groupby(['zona','articulo']).agg({'venta_kg':'sum'}).sort_values(by=['zona','venta_kg'],ascending=False).reset_index()


for n in niveles_zon:
    #se seleccionan los top 30 articulos por zona
    for a in list(t[t.zona==n].head(30).articulo.unique()):
      
        train = df_reg[(df_reg['articulo'] == a) & (df_reg['fecha'] < fecha_desde) & (df_reg[f'{nivel}'] == n)]
        test = df_reg[(df_reg['articulo'] == a) & (df_reg['fecha'] >= fecha_desde) & (df_reg['fecha'] <= fecha_hasta) & (df_reg[f'{nivel}'] == n)] #esto es el output de prophet y luego lo regulo con la regresion. porque a futuro no tengo otros elementos para la regresion salvo el output de prophet o sino tengo que hacer un multiple step ts
        if len(train['venta_kg']) > 3:
            X = train[cols]
            y = train[['venta_kg']]
            # entrenamiento --------
            regr = LinearRegression()
            regr.fit(X, y)
            del(X)
            del(y)
            print('fit ok')
   
            # pronóstico ---------
            if test.shape[0] >= 1:
                    test['pronostico'] = regr.predict(test[cols]) #luego el testcols tiene que darse como escenario. para estimar un fcst pero a 7 dias
 
            test_regrmult=pd.concat([test_regrmult,train,test]).drop_duplicates()   #,test     

test_regrmult=corregir_pronostico(test_regrmult)
test_regrmult.to_csv(pathd+f"/reg_mod/reg_{h}_Prophet_zona_futuro_all.csv", index=False) 




print(" ")
print(" ")
print(" ")
print(" ")
print(" ")
print("****************************************************************** ")
print("Termina regresion multiple")
print("****************************************************************** ")
print("Empezó: ", hora_empieza, " Terminó", datetime.today())
print("Tiempo de computo: ", (datetime.today()-hora_empieza))
print("****************************************************************** ")
print(" ")
print(" ")
print(" ")
print(" ")
print(" ")




