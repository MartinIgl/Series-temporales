#%%================================================================================================
# CARGA LIBRERIAS Y SCRIPTS NECESARIOS
#==================================================================================================

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta, date, datetime
import sys
import time
import parametros as param
from prophet import Prophet
import gc

import os
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



import snowflake.connector
con = snowflake.connector.connect()
# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

from operators.snowflake.warehouse import SnowflakeOperator
snowflake = SnowflakeOperator( )
#%% DEFINE PROCESO

#engine = create_engine('postgresql://bizm:osvaldo75@localhost:5432/bizm')  

#script = 'Procesa_Modelo.py'
#run =snowflake.df_from_query("SELECT MAX(run) as run FROM logs")+ 1
#run = run.run.unique()[0]
#%% DEFINE PROCESO

# script = 'procesa_VB.py'
#run =snowflake.df_from_query("SELECT MAX(run) as run FROM logs;")+ 1
#run = run.run.unique()[0]
# run=1
#%%================================================================================================
# LOGGER : crea tabla de logs
#==================================================================================================
#%%================================================================================================
# LOGGER : crea tabla de logs
#==================================================================================================
# def logger(f,df,c):
#     t0 = pd.to_datetime("now")
#     funcion = f.__name__
#     try:
#         d0 = f(df)
#         r = 'OK'
#         regs = d0.shape[0]
#     except:
#         r = 'NO OK'
#         regs = 0
#     t1 = pd.to_datetime("now")
#     tt = t1-t0
#     tt = ':'.join(str(tt).split(':')[:4])
#     log = {'run': [run] ,'subrun':[c], 'script': [script] ,'funcion': [funcion] ,'resultado':[r],'inicia': [t1],'duracion': [tt],'registros':[regs]}
#     log = pd.DataFrame(log)
#    log['inicia']=log['inicia'].dt.strftime('%Y-%m-%d %H:%M:%S')
# GRABAR LOG
#     engine = create_engine('postgresql://bizm:osvaldo75@localhost:5432/bizm')
#     log.to_sql('logs', engine, if_exists="append", index=False)
#     return d0



#%%================================================================================================
# lista upc
#==================================================================================================

def listado_upc(fecha_f,fecha_i,arg2=None,arg3=None):
    #Se define que upcs se van a forecastear
    #por ejemplo queremos forecastear los 5000 articulos de mayor venta en unidades en los ultimos x meses (previamente a quien llamaba df0):
    
    #1)genero la query de upc  para todos los locales 
    #NOTA: podriamos incluso parametrizar fecha y cantidad. el limite es mi n.

    
    q = """
                """.format(fecha_i,fecha_f)  
    
            
    a = snowflake.df_from_query(q) #creo la base de datos a partir de la tabla que pido con la query.
    #defino mi lista de upc para luego realizar los modelos 
    lista_upc = tuple(a.artc_artc_id.unique())
    del(q,a)
    gc.collect()
    return lista_upc


#%%================================================================================================
# QUERY y creacion de la base de datos para pronosticar
#==================================================================================================
##%%   
# HAY QUE AGRUPAR ACA POR ITEM PADRE#d = d[['fecha','upc','local','cantidad','stock','stock_minimo','precio','var_precio','flag_promo','flag_baja','flag_stock','flag_vta_basal','venta_basal']]


def query_y_creacion_DB_total(lista_upcs,arg1=None,arg2=None,arg3=None):
    #lista=tuple(lista_ip)
    #2) bajar los datos de venta basal de esos articulos:

 
    qvb = """
        """.format(ip=lista_upcs)
         


     df = snowflake.df_from_query(qvb).reset_index(drop='index').astype({'venta_basal': 'int'})


    df['grupo'] =np.where(df.desc_t.isin(['PERMANENTES BAS','ATEMPORALES BAS']), 'Permanentes','Temporada')

    df=df[df['artc_artc_id'].isin(lista_upcs)].reset_index(drop='index')
    

    df=df[['fecha',  'artc_artc_id','grupo','locales','venta_basal']]

    # GRABO en snowflake
    df.to_csv('-.csv', index=False)
    del(qvb)
    gc.collect()
    return df
    

#%%================================================================================================
# Modelo falta agregar el chunker acca y en los otros modelos
#==================================================================================================

def Modelo_WM_total(df, id_modelo):
    #Traigo la tabla de holidays (puede modificarse en la Base de datos)    
    H = """SELECT  * FROM --.FERIADOS"""
    Hol = snowflake.df_from_query(H) 


    gc.collect()
    #Traigo la tabla de parametros para tener los parameteros para el modelo
    p = """SELECT  * FROM PARAM_PROPHET"""
    df_param =    snowflake.df_from_query(p)
    param_grid=df_param.loc[df_param['id_modelo'] == id_modelo].drop(['id_modelo','denominacion'], axis=1).to_dict(orient='records')[0]

    #defino las fecha de cierre del local para hacer el filtro de ventas cero

    close1=list(Hol['ds'][Hol['holiday']=='cerrado'].astype(str))
    close2=list(Hol['ds'][Hol['holiday'].str.startswith('domingo', na=False)].astype(str))



    #genero la fecha de corte para diferenciar datos train y test (luego podra ser dado como input)
    inicial_fecha=(date.today()-timedelta(1)).strftime("%Y-%m-%d")
    fecha_fcts=(date.today()-timedelta(43)).strftime("%Y-%m-%d")

    
    def __datetime(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d')
    ultima_fecha=(__datetime(fecha_fcts)- timedelta(days = 600)).strftime("%Y-%m-%d")#2años
    


    dfout=pd.DataFrame(columns=['ds','yhat','yhat_upper','yhat_lower','artc_artc_id','locales','id_modelo']) #genero una estructura de dataframe con los datos con los que se guardarán los outputs del modelo
    dfout_temp=pd.DataFrame(columns=['ds','yhat','yhat_upper','yhat_lower','artc_artc_id','locales','id_modelo']) #genero una estructura de dataframe con los datos con los que se guardarán los outputs del modelo
    
    #Permanentes
    df0=df[df['grupo']=='Permanentes'].groupby(['fecha','artc_artc_id']).sum().drop(['locales'],axis=1).reset_index() #ordeno ip con mayor venta para todos los locales
    #Temporadas
    df1=df[df['grupo']=='Temporada'].groupby(['fecha','artc_artc_id']).sum().drop(['locales'],axis=1).reset_index() #ordeno ip con mayor venta para todos los locales

    #MODELO#Permanentes
    f=df0.artc_artc_id.unique()
    g=df1.artc_artc_id.unique()
    
    for ip in f: #recorro los item padre de la lista unica que había creado anteriormente en la creación de mi dato
        
        data=df0[df0['artc_artc_id']==ip].reset_index(drop='index') #recorro para cada ip
        #Selecciono la variable tiempo y var (variable requeridas por el modelo) y las defino con el nombre requerido
        data=data[['fecha','venta_basal']]
        data['fecha'] = pd.to_datetime(data['fecha'])
        data = data.rename(columns={'fecha': 'ds','venta_basal':'y'})
        
        #genero mis datos de train
        df_train = data[(data['ds']<=fecha_fcts)&(data['ds']>=ultima_fecha)]
        #Filtro los dias donde esta cerrado el local por cero de ventas.    
        #df_train.loc[(df_train['ds'].isin(close1)),'y'] =0 
        #df_train.loc[(df_train['ds'].isin(close2)),'y'] =0 

        #genero mis datos de test
        df_test = data[(data['ds']>fecha_fcts)&(data['ds']<=inicial_fecha)]
        #Filtro los dias donde esta cerrado el local por cero de ventas.    
        df_test.loc[(df_test['ds'].isin(close1)),'y'] =0 
        df_test.loc[(df_test['ds'].isin(close2)),'y'] =0 
        

        if len(df_train['ds'])<=2: #en caso de no pasar se saltea el proceso y se cambia de ip
               continue
        
        if len(df_test['ds'])<=2: #en caso de no pasar se saltea el proceso y se cambia de ip
               continue
        #instancio mi modelo con los parametros previamente seleccionados
        m = Prophet(**param_grid,holidays=Hol)
        
        #Entreno el modelo
        
        # used like
        #with suppress_stdout_stderr():
        m.fit(df_train) 
      
        #Genero mi predicción a futuro y los datos entrenados
        fitted=m.predict(df_train)
        #forecast = m.predict(df_test)
        future = m.make_future_dataframe(periods = 42, include_history = False,freq='D')
        forecast = m.predict(future)

        #Aplico filtro de cierre de local 
        fitted.loc[(fitted['ds'].isin(close1)),'yhat'] =0 
        fitted.loc[(fitted['ds'].isin(close1)),'yhat_lower'] =0 
        fitted.loc[(fitted['ds'].isin(close1)),'yhat_upper'] =0 
        forecast.loc[(forecast['ds'].isin(close1)),'yhat'] =0 
        forecast.loc[(forecast['ds'].isin(close1)),'yhat_lower'] =0 
        forecast.loc[(forecast['ds'].isin(close1)),'yhat_upper'] =0 
            
            
        fitted.loc[(fitted['ds'].isin(close2)),'yhat'] =0 
        fitted.loc[(fitted['ds'].isin(close2)),'yhat_lower'] =0 
        fitted.loc[(fitted['ds'].isin(close2)),'yhat_upper'] =0 
        forecast.loc[(forecast['ds'].isin(close2)),'yhat'] =0 
        forecast.loc[(forecast['ds'].isin(close2)),'yhat_lower'] =0 
        forecast.loc[(forecast['ds'].isin(close2)),'yhat_upper'] =0 
        
        ##filtro de valores negativos y de valores negativos (se reemplaza por el minimo valor de venta de la fecha a 4 semanas detrás)
        df_train['DIA_SEMANA'] =  pd.to_datetime(df_train.ds).dt.dayofweek 
        z = df_train[df_train['ds']>= str(pd.to_datetime(fecha_fcts) - timedelta(days = 30))].groupby('DIA_SEMANA')['y'].min().reset_index().rename(columns=({'y': 'min'}))
        forecast['DIA_SEMANA'] =  pd.to_datetime(forecast.ds).dt.dayofweek
        forecast = forecast.merge(z, on='DIA_SEMANA',how='left')
        forecast['yhat']=np.where((forecast['yhat']<=0),forecast['min'],forecast['yhat'])
        del(z,df_train,df_test) 

        forecast.loc[(forecast['yhat_lower'] <0),'yhat_lower']=0

        #genero mis datos de salida
        dfout1= pd.concat([fitted[['ds','yhat','yhat_upper','yhat_lower']], forecast[['ds','yhat','yhat_upper','yhat_lower']]])
        
        dfout1['artc_artc_id']=ip 
        #dfout1['id_modelo']=id_modelo
        #dfout1['locales']=float('nan')  #esta por la estructura de la tabla. falta el prorrateo. luego se saca esta parte
        #genero mi dataframe final 
        
        dfout=dfout.append(dfout1, ignore_index = True) 
        
    for ip in g: #recorro los item padre de la lista unica que había creado anteriormente en la creación de mi dato
        data=df1[df1['artc_artc_id']==ip].reset_index(drop='index') #recorro para cada ip
        #Selecciono la variable tiempo y var (variable requeridas por el modelo) y las defino con el nombre requerido
        data=data[['fecha','venta_basal']]
        data['fecha'] = pd.to_datetime(data['fecha'])
        data = data.rename(columns={'fecha': 'ds','venta_basal':'y'})
        
        #genero mis datos de train
        df_train = data[(data['ds']<=fecha_fcts)&(data['ds']>=ultima_fecha)]
        #Filtro los dias donde esta cerrado el local por cero de ventas.    
        #df_train.loc[(df_train['ds'].isin(close1)),'y'] =0 
        #df_train.loc[(df_train['ds'].isin(close2)),'y'] =0 

        #genero mis datos de test
        df_test = data[(data['ds']>fecha_fcts)&(data['ds']<=inicial_fecha)]
        #Filtro los dias donde esta cerrado el local por cero de ventas.    
        df_test.loc[(df_test['ds'].isin(close1)),'y'] =0 
        df_test.loc[(df_test['ds'].isin(close2)),'y'] =0 
        

        if len(df_train['ds'])<=2: #en caso de no pasar se saltea el proceso y se cambia de ip
               continue
        
        if len(df_test['ds'])<=2: #en caso de no pasar se saltea el proceso y se cambia de ip
               continue
        #instancio mi modelo con los parametros previamente seleccionados
        m = Prophet(**param_grid,holidays=Hol)
        
        #Entreno el modelo
        
        # used like
        #with suppress_stdout_stderr():
        m.fit(df_train) 
      
        #Genero mi predicción a futuro y los datos entrenados
        fitted=m.predict(df_train)
        #forecast = m.predict(df_test)
        future = m.make_future_dataframe(periods = 42, include_history = False,freq='D')
        forecast = m.predict(future)
        
        #Aplico filtro de cierre de local 
        fitted.loc[(fitted['ds'].isin(close1)),'yhat'] =0 
        fitted.loc[(fitted['ds'].isin(close1)),'yhat_lower'] =0 
        fitted.loc[(fitted['ds'].isin(close1)),'yhat_upper'] =0 
        forecast.loc[(forecast['ds'].isin(close1)),'yhat'] =0 
        forecast.loc[(forecast['ds'].isin(close1)),'yhat_lower'] =0 
        forecast.loc[(forecast['ds'].isin(close1)),'yhat_upper'] =0 
            
            
        fitted.loc[(fitted['ds'].isin(close2)),'yhat'] =0 
        fitted.loc[(fitted['ds'].isin(close2)),'yhat_lower'] =0 
        fitted.loc[(fitted['ds'].isin(close2)),'yhat_upper'] =0 
        forecast.loc[(forecast['ds'].isin(close2)),'yhat'] =0 
        forecast.loc[(forecast['ds'].isin(close2)),'yhat_lower'] =0 
        forecast.loc[(forecast['ds'].isin(close2)),'yhat_upper'] =0 
        
        ##filtro de valores negativos y de valores negativos (se reemplaza por el minimo valor de venta de la fecha a 4 semanas detrás)
        df_train['DIA_SEMANA'] =  pd.to_datetime(df_train.ds).dt.dayofweek 
        z = df_train[df_train['ds']>= str(pd.to_datetime(fecha_fcts) - timedelta(days = 30))].groupby('DIA_SEMANA')['y'].min().reset_index().rename(columns=({'y': 'min'}))
        forecast['DIA_SEMANA'] =  pd.to_datetime(forecast.ds).dt.dayofweek
        forecast = forecast.merge(z, on='DIA_SEMANA',how='left')
        forecast['yhat']=np.where((forecast['yhat']<=0),forecast['min'],forecast['yhat'])
        del(z,df_train,df_test) 


        forecast.loc[(forecast['yhat_lower'] <0),'yhat_lower']=0

        #genero mis datos de salida
        dfout2= pd.concat([fitted[['ds','yhat','yhat_upper','yhat_lower']], forecast[['ds','yhat','yhat_upper','yhat_lower']]])
        
        dfout2['artc_artc_id']=ip 
        #dfout2['id_modelo']=id_modelo
        #dfout2['locales']=float('nan')#esta por la estructura de la tabla. falta el prorrateo. luego se saca esta parte


        #genero mi dataframe final
        
        dfout_temp=dfout_temp.append(dfout2, ignore_index = True) 
    del(H,Hol,p,df_param,param_grid,close1,close2,df,df0,df1,f,g)
    gc.collect()
    return dfout,dfout_temp,fecha_fcts


### ================================================================================================
# prorrateo
#==================================================================================================
def Modelo_prorrateado(df,dfout,categoria):       #hay que chequear
    
    hoy=  date.today() 
    if categoria=='PERM':
        df=df[df['grupo']=='Permanentes']
    elif categoria=='TEMP':
        df=df[df['grupo']=='Temporada']
 
    #genero la variable para prorratear los upc seleccionados en cada local
    #forma 1: prorrateo de cada dia de cada upc en local ultimas 4 semanas

    pror=df[df['fecha']>=str(pd.to_datetime(hoy) - timedelta(days = 42))].groupby(['artc_artc_id','locales']).sum()/df[df['fecha']>=str(pd.to_datetime(hoy) - timedelta(days = 42))].groupby(['artc_artc_id']).sum()
    pror=pror.drop(['locales'],axis=1).reset_index() #revisar si esto me borra todos los locales
    pror=pror[pror.artc_artc_id.isin(df.artc_artc_id.unique())].rename(columns={'venta_basal': 'VB_prorrateada'})
    
    hoy=pd.to_datetime(hoy) - timedelta(days = 42)
    dfout1=dfout[dfout['ds']>hoy].drop(['locales','id_modelo'],axis=1).reset_index()

    zz=dfout1.merge(pror,on='artc_artc_id',how='outer')
    zz['yhat_pror_REAL']=zz['yhat']*zz['VB_prorrateada']
    zz['yhat_upper_pror_REAL']=zz['yhat_upper']*zz['VB_prorrateada']
    zz['yhat_lower_pror_REAL']=zz['yhat_lower']*zz['VB_prorrateada']
    zz=zz.drop(['VB_prorrateada','yhat',	'yhat_upper',	'yhat_lower'],axis=1).reset_index()
    zz=zz.rename(columns={'yhat_pror_REAL': 'yhat','yhat_upper_pror_REAL': 'yhat_upper','yhat_lower_pror_REAL': 'yhat_lower'})


    df_ph_pror=zz[['ds', 'yhat','yhat_upper','yhat_lower','artc_artc_id','locales']].reset_index(drop='index')
    df_ph_pror.reset_index(drop='index')
    
    del(df,dfout1,pror,zz)
    gc.collect()
    return df_ph_pror


#%%================================================================================================
# guardado WM total 
#==================================================================================================
def guarda_Tabla_FCST_pror(dfout_pror,start,categoria='PERM',arg2=None):
    #esta podría ser una variable o un parametro ejemplo  Fecha_pasada=dfout['ds'].iloc[-1] - timedelta(days = 60) #guardar los x dias de entrenamiento + los que se dieorn en el fcsts 
    #data=data[['fecha','venta_basal']]
        fecha_guardado=(date.today()-timedelta(142)).strftime('%Y-%m-%d') #toma la fecha de forecast+90 dias para atras

        #filtrarlo para que queden los datos del periodo forecasteado + los 60 dias previos al forecast
        df_final=dfout_pror[(dfout_pror['ds']>=datetime.strptime(fecha_guardado, "%Y-%m-%d"))]
        df_final['PRON_DIA_ID'] =start.strftime('%Y-%m-%d %H:%M:%S')
        df_final=df_final.rename(columns={"artc_artc_id":"ARTC_ARTC_ID",
                                        "ds": "TIEM_DIA_ID",
                                        "yhat": "PRONOSTICO", 
                                        "yhat_upper": "PRON_SUP", 
                                        "yhat_lower": "PRON_INF",
                                        "locales":"GEOG_LOCL_ID"})
        df_final = df_final[['TIEM_DIA_ID','GEOG_LOCL_ID','ARTC_ARTC_ID', 'PRONOSTICO','PRON_SUP','PRON_INF','PRON_DIA_ID']]

        
        df_final.to_csv(f'--/FCST_{categoria}.csv', index=False)
        if categoria=='PERM':
            #GRABO EN SNOWFLAKE
	(se graba a una tabla externa)

        elif categoria=='TEMP':

        con.close() 
        del(con)
        gc.collect()

