#%%
from prophet import Prophet
import pandas as pd
import numpy as np
import io
import base64
import streamlit as st
from multiprocessing import  Pool
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
#%%
print("Comienza a ejecutarse script fx_app.py", datetime.today())
pathd = "~/Proyectos/pronostico_carniceria/p2.0/datos"

#zonas = pd.read_csv(pathd+"/zonas_de_precio.csv")
#df_desc = pd.read_csv(pathd+"/item_desc_unit.csv") # descripcion de los artículos
feriados = pd.read_csv(pathd+"/feriados.csv")

#df_historico_base1 = pd.read_csv(pathd+"/base1_por_Suc.csv") 
#df_historico_base2 = pd.read_csv(pathd+"/base2_por_Suc.csv")
df_historico_fecha_max ='2022-03-01' #df_historico_base1.fecha.max()
df_historico_fecha_min ='2022-02-28'# df_historico_base1.fecha.min()

#df_excepciones = pd.DataFrame(columns=["fecha", "locales", "articulo", "precio", "pct_desc", "oferta"])


# Esto vamos a tener que tener cuidado. Si la fecha max es un miercoles entonces así:
fecha_min_a_pronosticar = pd.to_datetime(df_historico_fecha_max) + timedelta(days=1) 
fecha_max_a_pronosticar = pd.to_datetime(df_historico_fecha_max) + timedelta(days=28) 

# Esto vamos a tener que tener cuidado. Si la fecha max de historia es un jueves entonces así:
fecha_min_a_pronosticar = pd.to_datetime(df_historico_fecha_max)
fecha_max_a_pronosticar = pd.to_datetime(df_historico_fecha_max) + timedelta(days=27) 

# tambien tener cuidado con la funcion que descarga 3 semanas antes. si fecha min es un jueves entonces - 21, si es un miercoles entonces 20



#%%
################################################################################
######################### FUNCIONES ARMADO BASE #################################
################################################################################
def categorizar_ofertas(df):
    '''asigna si el tipo de precio corresponde a una oferta o no y si corresponde a una promo_volumen o no'''
    sin_oferta = ["E", "S", "C"]
    con_oferta = ["I", "O", "L"]
    conditions = [(df["tipo_precio"].isin(sin_oferta)), (df["tipo_precio"].isin(con_oferta))]
    values = ['no_oferta', 'oferta']
    df["tipo_precio"] = np.select(conditions, values, default = df["tipo_precio"])

    # promo_volumen
    # esto hay que ver como viene el input de ellos
    df["promo_volumen"] = df["promo_volumen"].fillna(0)
    df["promo_volumen"] = np.where(df["promo_volumen"] > 0, 1, 0)

    return df

def rename_variables(df):
    df = df.rename(columns={
        "CalendarDayID":"fecha",
        "LocID":"local",
        "ItemID":"art",
        "RetailItemQty":"venta_kg",
        "RetailTotalWithTaxAmt":"venta",
        "DiscountAmtWithTax":"desc",
        "DiscountPercentage":"pct_desc"})
    return df

#def transform_to_kg(df):
#    df = (df["ItemContent"] / 1000) * df["venta_kg"] 
#    return df

#def calculate_kg(df):
#    df["venta_kg"] = df.apply(lambda x: transform_to_kg(x) if x["MeasureUnitID"] == "GR" else x["venta_kg"], axis=1)
#    return df

def crear_feriados(df):
    """crea variables como feriado, cerrado, y feriado dia anterior y feriado dia posterior"""    
    # lo ideal sería tener una tabla que manejen ellos
    feriados = pd.read_csv(pathd+"/feriados.csv")
    
    feriados["fecha"] = pd.to_datetime(feriados["fecha"])
    df["fecha"] = pd.to_datetime(df["fecha"]) # lo ideal sería tener una tabla
    df = df.merge(feriados, how='left', on="fecha")

    df["feriado"] = df["feriado"].fillna(0).astype(int)
    df["cerrado"] = df["cerrado"].fillna(0).astype(int)

    df = df.sort_values(by=["fecha", "locales", "articulo"])

    df["feriado_ant"] = np.where(df["feriado"] == 1, 1, 0)
    df["feriado_ant"] = df.groupby(["locales", "articulo"])["feriado_ant"].shift(-1)

    df["feriado_pos"] = np.where(df["feriado"] == 1, 1, 0)
    df["feriado_pos"] = df.groupby(["locales", "articulo"])["feriado_pos"].shift(1)
    
    return df


def semana_desde_jueves(df):
    '''Fx repetida con calculo variables estacionales, pero en este caso crea una variable con nombre distinto'''
    df["fecha"] = pd.to_datetime(df['fecha'], format='%Y/%m/%d')
    df["semana"] = df["fecha"] - timedelta(days=3)  # agrego 3 dias to Mon to get to Thu 
    df["semana"] = df["semana"].apply(lambda x:x.isocalendar()[1])
    df["semana"] = df["semana"].map('semana {}'.format)
    return df

def crear_variables_estacionales(df):
    '''crea variables estacionales'''
    df["anio"] = df["fecha"].dt.year
    df["mes_anio"] = df["fecha"].dt.month_name()
    df = semana_desde_jueves(df)
    df["dia_semana"] = df["fecha"].dt.day_name()

    return df

def calcular_grupos_relativos(df):
    '''Crea los grupos de precios relativos, calcula el precio promedio del grupo por local-art-semana y luego el precio de cada art en relacion al grupo de precio que pertenece. En el caso de no pertenecer a ningun grupo, se divide sobre su mismo precio, dando como resultado '''
    #1. definición de grupos
    grupos_precios_relativos = pd.read_excel(pathd+"/grupos_precios_relativos.xlsx").rename(columns = {'art':'articulo'})

    grupos_precios_relativos["articulo"] = grupos_precios_relativos["articulo"].astype("int")
    

    df["articulo"] = df["articulo"].astype("int")
    df = df.merge(grupos_precios_relativos, how="left", on="articulo")

    # completo con el art en el caso que no tenga grupo
    df.loc[df['grupo'].isnull(),'grupo'] = df['articulo']

    #2. Precio promedio art y precio promedio grupo por semana

    df["precio_avg_art"] = df.groupby([pd.Grouper(key="fecha", freq="W-WED"), "locales", "articulo"])["precio"].transform('mean')
    df["precio_avg_grupo"] = df.groupby([pd.Grouper(key="fecha", freq="W-WED"), "locales", "grupo"])["precio"].transform('mean')

    #3. Precio relativo art vs grupo 
    df["precio_r"] = df["precio_avg_art"] / df["precio_avg_grupo"]
    #4 ordeno dataframe antes de comenzar a calcular desvios

    df = df.sort_values(by=["fecha", "locales", "articulo"])
    return df

def desvios_pr(data, var, periodo):
    ''' calcula el desvio del precio relativo sobre el promedio movil del precio relativo 
    en un periodo determinado por días (parametro periodo)'''
    data[f"pm_{var}{periodo}"] = data.groupby(["locales", "articulo"])[var].transform(lambda s: s.rolling(periodo, min_periods=1).mean().shift(1))
    
    data[f"dpm_{var}{periodo}"] = (data[var] / data[f"pm_{var}{periodo}"]).round(2)
    return data



# 10) Funciones para calcular tendencias en la venta 



def reemplazar_ceros(data, variable, dividendo, divisor):
    data[variable] = np.where((data[dividendo] == 0) & (data[divisor] == 0), 0, data[variable])
    return data
def reemplazar_inf(data, variable, q_95):
    data[variable] = np.where(data[variable].isin([np.inf]), q_95, data[variable])
    return data

# Tendencia kg dia de la semana
# Esta funcion es distinta en el script de base.py y tiene que ver con evitar usar datos del futuro.

def calcular_tend_dia_kg(data, n_dias):
    ''' Calcula el cociente entre: los kg vendidos el viernes previo (ej si presente es viernes) respecto de la mediana de kg vendidos los 4 viernes previos.'''

    #calcula la mediana de de los últimos n "viernes"
    data[f"mediana_dia_kg"] = data.groupby(["locales", "articulo", "dia_semana"])["venta_kg"].transform(lambda s: s.rolling(n_dias, min_periods=1).median().shift(1))

    #division entre venta de ese día y de la mediana de los últimos n "viernes"
    data[f"tend_dia_kg"] = data["venta_kg"] / data[f"mediana_dia_kg"]

    # reemplazo ceros
    data = reemplazar_ceros(data, "tend_dia_kg", "venta_kg", "mediana_dia_kg")

    #reemplazo inf. x = q_95
    q_95 = data.groupby(["locales", "articulo", "dia_semana"])["venta_kg"].transform(lambda x: x.quantile(0.95))
    data = reemplazar_inf(data, "tend_dia_kg", q_95)

    # donde se vendió igual o más kg que q_95, queda en tendencia_dia_kg q_95
    data[f"tend_dia_kg"] = np.where(data["venta_kg"] >= q_95, q_95, data["tend_dia_kg"])

    data = data.drop(f"mediana_dia_kg", axis=1)
 
    return data
 


# Tendencia semanal 
def calcular_median_tend_semanal_kg(data, n_semanas):
    '''Calcula el cociente entre los kilos de x semanas anteriores por el promedio de kg vendidos en x cantidad de semanas previas
    n = cantidad de semanas promedio a comparar. ahora 4.
    ''' 
    #suma kg de la semana
    data["kg_semana"] = data.groupby([pd.Grouper(key="fecha", freq="W-WED"), "anio", "semana", "locales", "articulo"])["venta_kg"].transform("sum")

    data = data[["anio", "semana", "locales", "articulo", "kg_semana"]].drop_duplicates()

    #promedio de los kg de las últimas n semanas
    data["kg_median_semanas"] = data.groupby(["locales", "articulo"])["kg_semana"].transform(lambda s: s.rolling(n_semanas, min_periods=1).median().shift(1))

    data[f"tend_median_semanal_kg"] = data["kg_semana"] / data["kg_median_semanas"]

    #reemplazo ceros
    data = reemplazar_ceros(data, "tend_median_semanal_kg", "kg_semana", "kg_median_semanas")

    #reemplazo inf 
    q_95 = data.groupby(["locales", "articulo"])["kg_semana"].transform(lambda x: x.quantile(0.95))
    data = reemplazar_inf(data, "tend_median_semanal_kg", q_95)
   
    return data[["anio", "semana","locales", "articulo", "tend_median_semanal_kg"]]

def calcular_tend_semanal_kg(data, n_semanas):
    '''Calcula el cociente entre los kilos de x semanas anteriores por el promedio de kg vendidos en x cantidad de semanas previas
    n = cantidad de semanas promedio a comparar. ahora 4.
    ''' 
    #suma kg de la semana
    data["kg_semana"] = data.groupby([pd.Grouper(key="fecha", freq="W-WED"), "anio", "semana", "locales", "articulo"])["venta_kg"].transform("sum")

    data = data[["anio", "semana", "locales", "articulo", "kg_semana"]].drop_duplicates()

    #promedio de los kg de las últimas n semanas
    data["kg_avg_semanas"] = data.groupby(["locales", "articulo"])["kg_semana"].transform(lambda s: s.rolling(n_semanas, min_periods=1).mean().shift(1))

    data[f"tend_semanal_kg"] = data["kg_semana"] / data["kg_avg_semanas"]

    #reemplazo ceros
    data = reemplazar_ceros(data, f"tend_semanal_kg", "kg_semana", "kg_avg_semanas")

    #reemplazo inf 
    q_95 = data.groupby(["locales", "articulo"])["kg_semana"].transform(lambda x: x.quantile(0.95))
    data = reemplazar_inf(data, f"tend_semanal_kg", q_95)

    return data[["anio", "semana", "locales", "articulo", "tend_semanal_kg"]]







def correcion_variable_tend(df, var):
    '''Fx que tiene en cuenta cuál es la semana que se va a pronosticar. Si corresponde a la primer semana, la variable tend_dia_kg toma el valor de la variable tend_dia_kg_1. Si se pronostica a dos semanas, la variable toma el valor de la variable tend_dia_kg_2. De esta forma, solo la variable tend_dia_kg entra entre los predictores y nos aseguramos de no utilizar una variable construida con datos del futuro.'''

    df["semana"] = df["fecha"] - timedelta(days=3) 
    df["semana"] = df["semana"].apply(lambda x:x.isocalendar()[1])
    df['semana_a_pronosticar'] = df['semana'].rank(method='dense', ascending=True)

    df[f"{var}"] = 0

    for i in range(1, df["semana_a_pronosticar"].nunique()+1):
        df[f"{var}"] = np.where(df['semana_a_pronosticar'] == i, df[f"{var}_{i}"], df[f"{var}"])
    
    df = df.drop([f"{var}_1", f"{var}_2", f"{var}_3", f"{var}_4", "semana", "semana_a_pronosticar"], axis=1)
    return df


def venta_min_max(df):
     '''calcula la venta minima y maxima de cada art-local-dia de la semana-mes. Se utiliza para corregir los pronosticos '''

     # df = df.sort_values(by=["fecha", "zona", "art"])

     df["venta_minima_kg"] = df.groupby(['zona', 'art', 'dia_semana', 'mes_anio'])["venta_kg"].transform('min')

     df["venta_minima_kg"] = df.groupby(["zona", "art", "mes_anio"], sort=False)["venta_minima_kg"].apply(lambda x: x.fillna(x.mean()))

     df["venta_minima_kg"].fillna(df.groupby(["zona", "art", "mes_anio"])["venta_kg"].transform('min'), inplace = True)
     df["venta_minima_kg"].fillna(df.groupby(["zona", "art"])["venta_kg"].transform('min'), inplace = True)

     df["venta_maxima_kg"] = df.groupby(['zona', 'art', 'dia_semana', 'mes_anio'])["venta_kg"].transform('max')

     df["venta_maxima_kg"].fillna(df.groupby(["zona", "art", "mes_anio"])["venta_kg"].transform('max'), inplace = True)

     df["venta_maxima_kg"].fillna(df.groupby(["zona", "art"])["venta_kg"].transform('max'), inplace = True)

     return df

################################################################################
####################### FUNCIONES PARA PROPHET ###############################
################################################################################

def calcular_prophet(df, param_grid, nivel,desde,hasta,fin):
    #feriados
    # df Hol con feriados
    feriado = df[['feriado','fecha','feriado_pos','feriado_ant','cerrado']]
    Hol = feriado[(feriado.feriado == 1)|(feriado.cerrado == 1)].drop_duplicates().rename(columns={"fecha": "ds",'feriado':'holiday', 'feriado_ant':'lower_window' ,'feriado_pos':'upper_window'})
    Hol.lower_window = np.where(Hol.lower_window == 1, -1, 0)
    Hol.holiday = np.where(Hol.cerrado==1,'cerrado','feriado')
    Hol.drop(columns='cerrado',inplace=True,axis=1)
    Hol.reset_index(drop='index')
    # para filtrar valores por dias cerrado
    close = Hol[Hol.holiday == 'cerrado']['ds'].to_list()
    
    # Tabla outputs
    dfouts = pd.DataFrame(columns=['ds','yhat','yhat_upper','yhat_lower','articulo', f'{nivel}'])
    # dfout_loc = pd.DataFrame(columns=['ds','yhat','yhat_upper','yhat_lower','art', f'{nivel}'])
    
    df = df[['fecha','articulo', f'{nivel}', 'venta_kg']]
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.rename(columns={'fecha': 'ds', 'venta_kg':'y'})
  
  
    
    if nivel =='locales':

        for n in df[f'{nivel}'].unique(): #vendria del merge
            tr = df[df[f'{nivel}'] == n]   
            te = df[df[f'{nivel}'] == n]
            for a in df.articulo.unique():
                train = tr[(tr['articulo'] == a) & (tr['ds'] < desde)&(tr['ds']>fin)][['ds','y']]
                #test = te[(te['articulo'] == a) & ((te['ds'] >= desde) & (te['ds'] <= hasta))][['ds','y']]
                #train = tr[(tr['articulo'] == a) & (tr['ds'] < desde)&(tr['ds']>fin)][['ds','y']]
                #train = tr[(tr['articulo'] == a) & (tr['ds'] <= hasta)&(tr['ds']>fin)][['ds','y']]
                if len(train['ds']) <= 2: #en caso de no pasar se saltea el proceso y se cambia de upc
                    continue
                
                # Instancio modelo con hiperparametros seleccionados (en este caso los holidays deben ponerse manualmente.)
                m = Prophet(**param_grid, holidays = Hol)
                # Entreno el modelo 
                m.fit(train) 
                # Genero mi predicción a futuro y el ajuste
                fitted = m.predict(train)
                #forecast = m.predict(test)
                
                # Genero corrección (es por local y art): reemplazo los valores negativos del forecast por el minimo valor de la serie de datos en las ultimas 4 semanas
                fitted['yhat'] = np.where(fitted['ds'].isin(close), 0, fitted.yhat.values)
                fitted['yhat_lower'] = np.where(fitted['ds'].isin(close), 0, fitted.yhat_lower.values)
                fitted['yhat_upper'] = np.where(fitted['ds'].isin(close), 0, fitted.yhat_upper.values)
                #Genero corrección (es por art para todos los locales): reemplazo los valores negativos del forecast por el minimo valor de la serie de datos en las ultimas 4 semanas            
                forecast['yhat'] = np.where(forecast['ds'].isin(close), 0, forecast.yhat.values)
                forecast['yhat_lower'] = np.where(forecast['ds'].isin(close), 0, forecast.yhat_lower.values)
                forecast['yhat_upper'] = np.where(forecast['ds'].isin(close), 0, forecast.yhat_upper.values)
                
                # Filtro de valores negativos y de valores negativos (se reemplaza por el minimo valor de venta de la fecha a 4 semanas detrás)
                train['dia_semana'] = train.ds.dt.dayofweek
                z = train[train['ds']>= str(pd.to_datetime(desde) - timedelta(days = 31))].groupby('dia_semana')['y'].min().reset_index().rename(columns=({'y': 'min'}))
                forecast['dia_semana'] = forecast.ds.dt.dayofweek
                forecast = forecast.merge(z, on='dia_semana',how='left')
                forecast['yhat']=np.where((forecast['yhat']<=0),forecast['min'],forecast['yhat'])
                del(z)
                
                
                forecast.loc[(forecast['yhat_lower'] < 0),'yhat_lower'] = 0
                #genero mi salida e incluyo los valores de loc y upc para cada salida
                dfout = pd.concat([fitted[['ds','yhat','yhat_upper','yhat_lower']], forecast[['ds','yhat','yhat_upper','yhat_lower']]])
                # print(dfout)
                dfout['articulo'] = a
                dfout[f'{nivel}'] = n
                dfouts = dfouts.append(dfout)
        # dfout_loc = dfout_loc.append(dfouts)
    elif nivel=='zona':
        
        df0=df.groupby(['ds','articulo','zona']).sum().reset_index()
        for n in df[f'{nivel}'].unique(): #vendria del merge
            print(n)
            #tr = df0[df0[f'{nivel}'] == n]
            #te = df0[df0[f'{nivel}'] == n]
            
            for a in list(df[df.zona==n].head(70).articulo.unique()):
                print(a)
                #train = tr[(tr['articulo'] == a) & (tr['ds'] <= desde)&(tr['ds']>fin)][['ds','y']]
                train =df0[(df0['articulo'] == a)&(df0[f'{nivel}'] == n)&(df0['ds'] < desde)&(df0['ds']>fin)][['ds','y']]
                #test = te[(te['articulo'] == a) & ((te['ds'] >= desde) & (te['ds'] <= hasta))][['ds','y']]
                # El train tiene que ser por cada art todo lo anterior a desde_test


                if len(train['ds']) <= 2: #en caso de no pasar se saltea el proceso y se cambia de upc
                    continue
                
                if len(train['ds']) <= 2: #en caso de no pasar se saltea el proceso y se cambia de upc
                    continue
                  
                # Instancio modelo con hiperparametros seleccionados (en este caso los holidays deben ponerse manualmente.)
                m = Prophet(**param_grid, holidays = Hol)
                # Entreno el modelo 
                m.fit(train) 
                # Genero mi predicción a futuro y el ajuste
                fitted = m.predict(train)
                #forecast = m.predict(test)
                future =  m.make_future_dataframe(periods = 8 , freq='D')
                forecast = m.predict(future)
                
                    # Genero corrección (es por local y art): reemplazo los valores negativos del forecast por el minimo valor de la serie de datos en las ultimas 4 semanas
                fitted['yhat'] = np.where(fitted['ds'].isin(close), 0, fitted.yhat.values)
                fitted['yhat_lower'] = np.where(fitted['ds'].isin(close), 0, fitted.yhat_lower.values)
                fitted['yhat_upper'] = np.where(fitted['ds'].isin(close), 0, fitted.yhat_upper.values)
                    #Genero corrección (es por art para todos los locales): reemplazo los valores negativos del forecast por el minimo valor de la serie de datos en las ultimas 4 semanas            
                forecast['yhat'] = np.where(forecast['ds'].isin(close), 0, forecast.yhat.values)
                forecast['yhat_lower'] = np.where(forecast['ds'].isin(close), 0, forecast.yhat_lower.values)
                forecast['yhat_upper'] = np.where(forecast['ds'].isin(close), 0, forecast.yhat_upper.values)
                    
                    # Filtro de valores negativos y de valores negativos (se reemplaza por el minimo valor de venta de la fecha a 4 semanas detrás)
                train['dia_semana'] = train.ds.dt.dayofweek
                z = train[train['ds']>= str(pd.to_datetime(desde) - timedelta(days = 31))].groupby('dia_semana')['y'].min().reset_index().rename(columns=({'y': 'min'}))
                forecast['dia_semana'] = forecast.ds.dt.dayofweek
                forecast = forecast.merge(z, on='dia_semana',how='left')
                forecast['yhat']=np.where((forecast['yhat']<=0),forecast['min'],forecast['yhat'])
                del(z)
                    
                    
                forecast.loc[(forecast['yhat_lower'] < 0),'yhat_lower'] = 0
                #genero mi salida e incluyo los valores de loc y upc para cada salida
                dfout = pd.concat([fitted[['ds','yhat','yhat_upper','yhat_lower']], forecast[['ds','yhat','yhat_upper','yhat_lower']]])
                # print(dfout)
                dfout['articulo'] = a
                dfout[f'{nivel}'] = n
                dfouts = dfouts.append(dfout)
                


    return dfouts
        

################################################################################
####################### FUNCIONES PARA REGRESION ###############################
################################################################################

def calcular_regresion(data, fecha_desde, fecha_hasta, cols, nivel='zona'):
    test_regrmult = pd.DataFrame()
    # train_regrmult = pd.DataFrame()
    art = data.art.unique().tolist()
    niveles = data[nivel].unique().tolist()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    for a in art:
        tr = data[(data['art'] == a) & (data['fecha'] < fecha_desde)]
        te = data[(data['art'] == a) & ((data['fecha'] >= fecha_desde) & (data['fecha'] <= fecha_hasta))]

        for n in niveles:
            train = tr[tr[f'{nivel}'] == n]
            test = te[te[f'{nivel}'] == n]

            cols_to_remove = test.columns[test.isna().any()].tolist()
            cols = [ x for x in cols if x not in cols_to_remove ]
            print(a, n, 'cols to remove:', cols_to_remove)

            if train[train['venta_real'] > 0].shape[0] > 20:
                X = train[cols]
                y = train[['venta_real']]

                # entrenamiento --------
                regr = LinearRegression()
                regr.fit(X, y)

                # # pronóstico del train set-------
                # train['pronostico'] = regr.predict(train[cols])
                # train_regrmult = pd.concat([train_regrmult, train])
                # del(train)

                # pronóstico ---------
                test = te[te[f'{nivel}'] == n]
                #print('art:'+str(a)+' | local:'+str(l))
                if test.shape[0] >= 1:
                    test['pronostico'] = regr.predict(test[cols])
                    test_regrmult = pd.concat([test_regrmult, test])
                    del(test)
                del(X)
                del(y) 

    return test_regrmult

def corregir_pronostico(data, limite):
    data["fecha"] = pd.to_datetime(data['fecha'], format='%Y/%m/%d')
    data['pronostico'] = np.where(data['pronostico'] <= 0, data['venta_minima_kg'], data['pronostico'])
    data['pronostico'] = np.where((data["pronostico"] > (data["venta_maxima_kg"] * limite)), data["venta_maxima_kg"], data["pronostico"])
    return data


################################################################################
####################### FUNCIONES PARA STREAMLIT ###############################
################################################################################

def joinear_listas_precios(listas_precios):
    ''' En el caso de que el sector suba más de una lista de precio, es decir, calcule más de una zona de precio a la vez, appendea las listas de precios en una única lista de precios.'''
    precios = pd.DataFrame(columns=['zona','art','precio','pct_desc'])
                
    for lista_precios in listas_precios:
        #Lee la nueva lista de precios y rellena en caso de NaN
        lista_precios = pd.read_excel(lista_precios)
        lista_precios["pct_desc"] = lista_precios["pct_desc"].fillna(0)
        precios = precios.append(lista_precios, ignore_index=True)

    return precios

# Funciones para descargar excels 

def download_excel(object_to_download, intro, fd, fh, download_link_text):
    
    fd = fd.strftime("%m-%d-%Y")
    fh = fh.strftime("%m-%d-%Y")
    file_name = f"{intro}_{fd}_{fh}.xlsx"
    
    towrite = io.BytesIO()
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=False, header=True)

    towrite.seek(0)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    # return f'<a href="data:file/txt;base64,{b64}" download={object_to_download.to_excel("{intro}_{zona}_{fd}_{fh}.xlsx")}>{download_link_text}</a>'
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{file_name}.xlsx">{download_link_text}</a>'


def download_excel2(object_to_download, file_name, download_link_text):
    
    towrite = io.BytesIO()
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=False, header=True)

    towrite.seek(0)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    # return f'<a href="data:file/txt;base64,{b64}" download={object_to_download.to_excel("{intro}_{zona}_{fd}_{fh}.xlsx")}>{download_link_text}</a>'
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{file_name}.xlsx">{download_link_text}</a>'


def crear_lista_precios(zonas, precios):
    '''Multiplica las zonas-art-precio por cada fecha que deba pronosticarse. 
    En este caso, arma el pronostico para los proximos 28 días desde el último dato disponible '''

    y1 = pd.date_range(pd.to_datetime(fecha_min_a_pronosticar), pd.to_datetime(fecha_max_a_pronosticar), freq="D")
    y1 = pd.DataFrame({"fecha" : y1})
    y2 = df_historico_base1[df_historico_base1.zona.isin(zonas)][["zona", "art"]].drop_duplicates()
    df_pronostico = y1.assign(dummy=1).merge(y2.assign(dummy=1), on='dummy', how='outer').drop('dummy', axis=1)
    del(y1, y2)
    df_pronostico = df_pronostico.merge(precios, how="left", on=["zona", "art"])
    return df_pronostico


def actualizar_excepciones_precios(df_pronostico, precios_excepciones):
    '''En el caso qu un día específico de la semana quieran poner un precio excepcional, esta función actualiza esa fecha específica'''
    df_excepciones = pd.read_excel(precios_excepciones)

    df_pronostico["fecha"] = pd.to_datetime(df_pronostico["fecha"])
    df_excepciones["fecha"] = pd.to_datetime(df_excepciones["fecha"])

    df_pronostico = df_excepciones.set_index(["fecha", "zona", "art"]).combine_first(df_pronostico.set_index(["fecha", "zona", "art"])).reset_index()

    return df_pronostico


def descargar_historia(df, zonas, fd, fh):
    df["fecha"] = pd.to_datetime(df['fecha'], format='%Y/%m/%d')
    df = df[df.zona.isin(zonas)]
    df = df[(df.fecha >= fd) & (df.fecha <= fh)]

    return df

def armar_df_pronostico(df):
    df = semana_desde_jueves(df) # para que aparezca el nº de la semana
    df["art"] = df["art"].astype("int")
    df["pronostico"] = df["pronostico"].round(2)
    df["precio"] = df["precio"].round(2)
    df = df.merge(df_desc, how="left", on="art") # descripcion art
    df = df.groupby(["zona", "semana", "art", "descripcion", "unidad"]).agg({"precio":"mean", "pronostico":"sum"}).reset_index() #suma la cantidad de kilos de esos días/semanas

    df = df[["zona", "semana", "art", "descripcion", "unidad", "precio", "pronostico"]]
    
    return df

def armar_df_3_semanas_previas(pronostico, zonas, fd, fh):
    # descargo historia hasta < fecha_min_a_pronosticar
    df_historico = descargar_historia(df_historico_base1, zonas, fd, fh)
    
    # desde 20 dias atras (3 semanas)
    df_historico = df_historico[df_historico.fecha >= (pd.to_datetime(fecha_min_a_pronosticar) - timedelta(days=21))]
    df_historico = semana_desde_jueves(df_historico)

    # variables para describir semanas de historias que trae esta función
    df_historico_semana_min = df_historico.semana.min()
    df_historico_fecha_semana_min = df_historico.fecha.min()
    df_historico_semana_max = df_historico.semana.max()
    df_historico_fecha_semana_max = df_historico.fecha.max()

    # df_historico = df_historico.merge(zonas_seleccionadas, how="left", on="local")
    df_historico = df_historico.merge(df_desc, how="left", on="art")

    # pivot kilos x semana
    df_historico = calcular_kg_con_sin_oferta(df_historico)

    # pivot pronostico x semana
    p2 = pronostico.copy()
    p2['N'] = p2.groupby(["zona", "art", "descripcion", "unidad", "semana"]).cumcount()
    p2 = p2.pivot_table(index=["zona", "art", "descripcion", "unidad", "N"], columns="semana", values=["precio", "pronostico"],aggfunc='first')
    p2.columns = [x[1] if x[0] == '' else ' '.join(x[::-1]) for x in p2.columns]
    p2 = p2.sort_index(axis=1).reset_index().drop(columns='N')
    p2 = p2.round(2)

    df = df_historico.merge(p2, how="outer", on=["zona", "art", "descripcion", "unidad"])
    num = df.select_dtypes(include=[np.number])
    df.loc[:, num.columns] = np.round(num, 2)

    return df, df_historico_semana_min, df_historico_fecha_semana_min, df_historico_semana_max, df_historico_fecha_semana_max


def calcular_kg_con_sin_oferta(df):
    # kg vendidos con oferta y sin oferta
    # df["oferta"] = np.where((df["pct_desc"] == 0) | (df["tipo_precio"] == "no_oferta"), "sin oferta", "oferta")
    df["oferta"] = np.where((df["pct_desc"] == 0) , "sin oferta", "oferta")

    df = df.groupby([pd.Grouper(key="fecha", freq="W-WED"), "semana", "zona", "art", "descripcion", "unidad", "oferta"]).agg({"venta_kg": "sum", "precio":"mean"}).reset_index() 
    
    #kg vendidos en total
    df["venta kg total"] = df.groupby([pd.Grouper(key="fecha", freq="W-WED"), "semana", "zona", "art", "descripcion", "unidad"])["venta_kg"].transform("sum")

    # pivot kg con oferta, sin oferta
    df = df.pivot_table(index=["semana", "zona", "art", "descripcion", "unidad", "precio", "venta kg total"], columns="oferta", values="venta_kg").reset_index()

    df[["oferta", "sin oferta"]] = df[["oferta", "sin oferta"]].fillna(0)

    df = df.rename(columns={"oferta": "venta kg oferta", "sin oferta": "venta kg sin oferta"})

    df = df.rename_axis(None, axis=1).reset_index(drop=True)

    # pivot semanas y ventas
    df['N'] = df.groupby(["zona", "art", "descripcion", "unidad", "precio", "semana"]).cumcount()

    df1 = df.pivot_table(index=["zona", "art", "descripcion", "unidad", "N"], 
                        columns="semana", 
                        values=["precio", "venta kg total", "venta kg oferta", "venta kg sin oferta"],
                        aggfunc='max')

    df1.columns = [x[1] if x[0] == ' ' else ' '.join(x[::-1]) for x in df1.columns]
    df1 = df1.sort_index(axis=1).reset_index().drop(columns='N')
    df1 = df1.round(2)

    return df1


def calcular_resumen_kg(fd, fh, zonas):
    df = df_historico_base1.copy()

    df["fecha"] = pd.to_datetime(df['fecha'], format='%Y/%m/%d')

    df = df[(df.fecha >= fd) & (df.fecha <= fh)]

    df = df[df.zona.isin(zonas)]

    df = df.merge(df_desc, how="left", on="art")
    
    df = semana_desde_jueves(df)

    return df

print("Finaliza script funciones_app.py", datetime.today())