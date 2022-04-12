# ==================================
# IMPORTAMOS LAS LIBRERIAS NECESARIAS
# ===================================

# -*- coding: utf-8 -*-

# -- Manejo de datos
import pandas as pd
import numpy as np
import ROOT as r
from read_data import *

# -- Para plottear
import matplotlib.pyplot as plt

# -- Librerias de ML
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# ==================================
# DEFINIMOS FUNCIONES PARA PROCESAR DATOS
# ===================================
   
def process_dataframe(df):
    '''
    En esta funcion procesamos la informacion de nuestro dataframe, que 
    por defecto viene con la informacion basica de las rootfiles. (e.g.
    el pT de los leptones viene en una sola variable "LepGood_pT" que
    es vectorial; a nosotros nos interesa tener tres columnas diferentes,
    cada una con el pT de los leptones que leemos. Todo este tipo de 
    logicas tienen que implementarse en esta funcion.
    '''
    entries = df.shape[0]

    # -- Ejemplo: Columnas de pT de los leptones
 
    # NOTA: Por defecto sabemos que todas nuestras filas deben tener 3 elementos 
    #       en la columna de LepGood_pt, pues tenemos exclusivamente 3 leptones!
    #       sabiendo esto, podemos usar listas de python para procesar los datos
    #       de manera eficiente.


    # ------------------------------------------------
    nLepGood = 3 # numero de leptones
    lep_pt_arr = df["LepGood_pt"].to_numpy() # Nos cogemos la columna de pT
    lep_pt_arr = np.hstack(lep_pt_col) # Esto aplana el array
    lep_pt_arr = lep_pt_arr.reshape((entries, nLepGood)) # Le cambiamos la forma al array

    # Creamos un nuevo DataFrame con las columnas separadas
    lep_pt_pd = pd.DataFrame(lep_pt_arr,
                          columns = ["lep1_pt", "lep2_pt", "lep3_pt"],
                          index = df.index)
    
    df.pop("LepGood_pt") # Quitamos la antigua columna 
    df = combine_dataframes([df, lep_pt_pd], axis = 1) # Incluimos la nueva columna
    # ------------------------------------------------
    return df

# ==== Aqui empieza la magia ====
# Esto de if __name__ es para que el compilador
# sepa donde empieza la logica de tu codigo. 
# Esta muy bien para mantener un codigo limpio
# en python.

if __name__ == "__main__":
    # -- Nos cargamos los datos en formato de pandas.dataframe
    mainpath = "/pool/phedex/TOPnanoAODv6/ttW_MC_Ntuples_skim/2016/"
    muestras = {"ttbar"      : ["TTLep_pow_part1"], 
                "ttW"        : ["TTWToLNu"]}

    # ======== CONFIGURACIONES ========== #
    # -- Variables que vamos a usar para procesar datos:
    vars_keep = ["LepGood_pt", "LepGood_eta", "nLepGood"]
    vars_friend_recl = ["nJet25_Recl"] 

    # -- Muestras que vamos a utilizar
    ttbar = muestras["ttbar"][0]
    ttw = muestras["ttW"][0]
    
    # -- Nos cargamos los dataframes con la funcion load_data (definida en read_data.py)
    df_ttw   = load_data(mainpath + ttw, vars_keep)
    df_ttw   = add_friends(df_ttw, mainpath+"1_recl_enero/"+ttw, vars_friend_recl)

    df_ttbar = load_data(mainpath + ttbar, vars_keep)
    df_ttbar = add_friends(df_ttbar, mainpath+"1_recl_enero/"+ttbar, vars_friend_recl)
   
    # Combinamos todos los dataframes
    dfs_to_combine = [df_ttw, df_ttbar]
    df = combine_dataframes(dfs_to_combine, axis = 0) # Para concatenar filas

    #  == Definimos nuestra region de senyal == 
    # -- Pedimos: 3 leptones 
    #             >=2 jets
    df_3l = df[ df["nLepGood"] ==3 ]
    df_3l = df[ df["nJet25"] >= 2 ]

    df_3l = process_dataframe(df_3l)
