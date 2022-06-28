#%%
#==============================================================================
# SCRIPT: procesa_venta_basal.py 
# VERSION: 0.3 (con snowflake)
#===========================================================================


# carga librerias y scripts necesarios ----------------------------------------
import numpy as np
import pandas as pd
import fx_VB_BAS as fx
import datetime
from datetime import timedelta, date, datetime
import warnings
from joblib import Parallel, delayed

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

    
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



print('empezo: '+date.today().strftime('%Y-%m-%d %H:%M:%S') )
start1= pd.to_datetime("now")


print('--------------------Inicio del proceso------------------------------')
import fx_ph as fxp


fecha_hoy=date.today()

#defino fechas fcst y entrenamiento
fecha_f=(fecha_hoy-timedelta(1)).strftime("%Y-%m-%d")
fecha_i=(fecha_hoy-timedelta(600)).strftime("%Y-%m-%d")  #conviene un año por el momento
fecha_hoy=fecha_hoy.strftime("%Y-%m-%d")

##############################

start= pd.to_datetime("now")
lista_upcs = fxp.listado_upc(fecha_f,fecha_i,0,0)
end= pd.to_datetime("now")
print('---------- fx.listado_upc ------------------------------')
print(':'.join(str(end - start).split(':')[:4]))
print('#Art_ID: '+str(len(lista_upcs)))



print('Inicio proceso  en paralelo de Prophet ')
idM=1
print('El modelo es el id: '+ str(idM))

n = 100000    
def Prophet_BAS(upcs,idM):
        ##########################Lectura############################
    start = pd.to_datetime("now")
    df = fxp.query_y_creacion_DB_total(upcs,0,0,0)
    end= pd.to_datetime("now")
    print('---------- fx.queries_y_procesa_datos --------------------')
    print(':'.join(str(end - start).split(':')[:4]))
    print(len(df.artc_artc_id.unique()))
    print(len(df.locales.unique()))
    print('')

    
    ##########################Modelo ############################
    #para todos los locales 
    start = pd.to_datetime("now")
    dfout,dfout_temp,fecha_fcts = fxp.Modelo_WM_total(df, idM)     #dfout es permanentes y el otro de temporada
    end= pd.to_datetime("now")    
    print('---------- fx.datos_Modelados_sum_loc --------------------')
    print(':'.join(str(end - start).split(':')[:4]))
    print(len(dfout.artc_artc_id.unique()))
    print(len(dfout.locales.unique()))
    print('fecha de FCST: '+str(fecha_fcts))
    print('')
    
    #Prorrateo 
    start = pd.to_datetime("now")
    dfout_pror_p   = fxp.Modelo_prorrateado(df,dfout,'PERM')
    print(len(dfout_pror_p.artc_artc_id.unique()))
    print(len(dfout_pror_p.locales.unique()))
    print('----')
    dfout_pror_t   = fxp.Modelo_prorrateado(df,dfout,'TEMP')
    print(len(dfout_pror_t.artc_artc_id.unique()))
    print(len(dfout_pror_t.locales.unique()))
    
    end= pd.to_datetime("now")    
    print('---------- fx.Modelo_prorrateado --------------------')
    print(':'.join(str(end - start).split(':')[:4]))
    print('')
    
    ##########################Grabado############################

    start = pd.to_datetime("now")
    datos_save = fxp.guarda_Tabla_FCST_pror(dfout_pror_p,start,'PERM',0)
    
    print('---------- fx.guardado_datos P--------------------')
    datos_save = fxp.guarda_Tabla_FCST_pror(dfout_pror_t,start,'TEMP',0)
    end= pd.to_datetime("now")
    print('---------- fx.guardado_datos T--------------------')
    print(':'.join(str(end - start).split(':')[:4]))
    
Parallel(n_jobs=3)(delayed(Prophet_BAS)(upcs,idM) for upcs in chunker(lista_upcs,n))



print('termino: '+date.today().strftime('%Y-%m-%d %H:%M:%S') )
end1= pd.to_datetime("now")
print('')
print('')
print('')
print('')
print('')
print('------------------------------------------------------------------')
print('----------------FIN DE PROCESO FCST --------------------------------')
print('Tiempo de proceso: '+':'.join(str(end1 - start1).split(':')[:4]))
print('------------------------------------------------------------------')
print('')
print('')
print('')
print('')
print('')

# %%
