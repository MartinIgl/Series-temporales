# app2.py
# %% LIBRERIAS ---------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from st_aggrid import AgGrid

st.set_page_config(layout="wide")
pd.set_option('display.max_columns', None)

from datetime import datetime, timedelta

def semana_desde_jueves(df):
    '''Fx repetida con calculo variables estacionales, pero en este caso crea una variable con nombre distinto'''
    df["fecha"] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
    df["semana"] = df["fecha"] - timedelta(days=3)  # agrego 3 dias to Mon to get to Thu 
    df["semana"] = df["semana"].apply(lambda x:x.isocalendar()[1])
    df["semana"] = df["semana"].map('{}'.format)
    return df

def crear_variables_estacionales(df):
    '''crea variables estacionales'''
    df["fecha"] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
    df = semana_desde_jueves(df)
    return df



path=''
t='m29'
#%% APP CLIENTES / PUNTOS
#p28 = pd.read_csv(path+'datos/pronostico_prophet_4_365_zona.csv')
d29 = pd.read_csv(path+f'datos/reg_mod/futuro_vs_sim_{t}_all_a.csv')
d29=d29[(d29.fecha>='2022-05-26')]
#%%
z = st.selectbox("Zona de Precios:", d29.zona.unique())

#%%
z29 = d29[(d29['zona']==z)]
t29 = z29.groupby(['articulo','zona']).agg({
    'venta_kg':'sum',
    'yhat':'sum',
    'pronostico':'sum',
    'venta_kg_simulado':'max'
}).reset_index()

#tt29 = z29.groupby(['zona']).agg({
#    'venta_kg':'sum',
#    'yhat':'sum',
#    'pronostico':'sum'
#}).reset_index()
#tt29['articulo'] = 'TODOS'
#tt29 = tt29[['articulo','zona','venta_kg','yhat','pronostico']]
#t29 = tt29.append(t29, ignore_index=True)

t29['Dif. Prophet'] = (t29['venta_kg'] - t29['yhat'])
t29['Dif. RM'] = (t29['venta_kg'] - t29['pronostico']) 
t29['Dif. RMvsSim'] = (t29['venta_kg_simulado'] - t29['pronostico']) 
t29['Desvio Prophet'] = (t29['venta_kg'] - t29['yhat']) /  t29['venta_kg']
t29['Desvio RM'] = (t29['venta_kg'] - t29['pronostico']) /  t29['venta_kg']
t29['Desvio Sim'] = (t29['venta_kg'] - t29['venta_kg_simulado']) /  t29['venta_kg']
t29 = t29.round({'yhat': 1, 
                 'venta_kg': 1,'venta_kg_simulado': 1,'pronostico': 1,
                 'Dif. Prophet':1,
                 'Dif. RM':1,'Dif. RMvsSim':1,
                 'Desvio Prophet':3,
                 'Desvio RM':3,'Desvio Sim':3})
t29 = t29[t29['venta_kg']>0]
del(t29['zona'])
t29.columns = ['Articulo','venta_kg','Pron.Prophet','Pron.RM',
               'venta_kg_simulado',
                'Dif.Prophet', 'Dif.RM','Dif. RMvsSim',
               'Desvio Prophet',
               'Desvio RM','Desvio Sim']
#creo objeto para gravicar
sc = make_subplots(rows=1, cols=3, shared_yaxes=True, vertical_spacing=0.02)
sc.add_trace(go.Scatter(x=t29['Pron.Prophet'], y=t29.venta_kg, mode='markers', name='Pron. Prophet'), row=1, col=1)
sc.add_trace(go.Scatter(x=t29['Pron.RM'], y=t29.venta_kg, mode='markers', name='Pron. Regr.Mult.'), row=1, col=1)
sc.add_trace(go.Scatter(x=t29.venta_kg_simulado, y=t29.venta_kg, mode='markers', name='simulado'), row=1, col=1)

sc.add_trace(go.Scatter(x=t29['Dif.Prophet'], y=t29.venta_kg, mode='markers', name='Dif. Prophet'), row=1, col=2)
sc.add_trace(go.Scatter(x=t29['Dif.RM'], y=t29.venta_kg, mode='markers', name='Dif. Regr.Mult.'), row=1, col=2)
sc.add_trace(go.Scatter(x=t29['Dif. RMvsSim'], y=t29.venta_kg, mode='markers', name='Dif. RM Vs Sim'), row=1, col=2)

sc.add_trace(go.Scatter(x=t29['Desvio Prophet'], y=t29.venta_kg, mode='markers', name='Desvio Prophet'), row=1, col=3)
sc.add_trace(go.Scatter(x=t29['Desvio RM'], y=t29.venta_kg, mode='markers', name='Desvio Regr.Mult.'), row=1, col=3)
sc.add_trace(go.Scatter(x=t29['Desvio Sim'], y=t29.venta_kg, mode='markers', name='Desvio Sim'), row=1, col=3)


sc.update_layout(yaxis_title="Venta Real en Kgrs.",width=1100, height=500)
#instruccion para graficar plotliy
st.plotly_chart(sc)



#%%
AgGrid(t29, editable=False)
t='m29'

path='/home/miglesias/Proyectos/pronostico_carniceria/p2.0/'
d29 = pd.read_csv(path+f'datos/reg_mod/futuro_vs_sim_{t}_a.csv')

d29 = crear_variables_estacionales(d29)


a = st.selectbox("Articulo:", d29.articulo.unique())
d29=d29[(d29.fecha>='2022-05-26')]
m29 = d29[(d29['articulo']==a)&(d29.zona==z)]

m29=m29.groupby(['articulo', 'zona', 'semana']).agg({'venta_kg':'sum', 'yhat':'sum', 'pronostico':'sum', 'venta_kg_simulado':'max'}).reset_index()
# %% GRAFICO -----------------------------------------------------------------------------
fig = make_subplots()
fig.add_trace(go.Scatter(x=m29.semana, y=m29.venta_kg, mode='lines+markers', name='Venta Real'), row=1, col=1)
fig.add_trace(go.Scatter(x=m29.semana, y=m29.pronostico, mode='lines+markers', name='Regr.Mult.'), row=1, col=1)
fig.add_trace(go.Scatter(x=m29.semana, y=m29.yhat, mode='lines+markers', name='Prophet'), row=1,col=1)
fig.add_trace(go.Scatter(x=m29.semana, y=m29.venta_kg_simulado, mode='lines+markers', name='sim'), row=1, col=1)

fig.update_layout(width=1100, height=350)
st.plotly_chart(fig)

     

# 
#fig = make_subplots()
#fig.add_trace(go.Scatter(x=m29.semana, y=m29.venta_kg, mode='lines+markers', name='Venta Real'), row=1, col=1)
#fig.add_trace(go.Scatter(x=m29.semana, y=m29.venta_kg_simulado, mode='lines+markers', name='sim'), row=1, col=1)
#fig.add_trace(go.Scatter(x=m29.semana, y=m29.pronostico, mode='lines+markers', name='Regr.Mult.'), row=1, col=1)
#fig.add_trace(go.Scatter(x=m29.semana, y=m29.yhat, mode='lines+markers', name='Prophet'), row=1,col=1)
#fig.update_layout(width=1100, height=350)
#st.plotly_chart(fig)
