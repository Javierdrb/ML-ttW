# -*- coding: utf-8 -*-

# -- Manejo de datos
import pandas as pd
import numpy as np
import ROOT as r
from root_numpy import tree2array

# -- Para quitar un warning muy pesado que salta por unas librerias de numpy
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.object` is a deprecated alias')

def combine_dataframes(dfs, axis = 0):
    '''
    Esta funcion combina varios dataframes:
      axis=0 -- se concatenan filas (i.e. metemos mas sucesos)
      axis=1 -- se concatenan columnas (i.e. metemos mas variables)
    '''
    ignore_index = True if axis == 0 else False
    super_df = pd.concat(dfs, axis, ignore_index=ignore_index) 
    return super_df

def load_data(file_, vars_ = [], treename = "Events", start = 0, stop = 1000): 
    '''
    Esta funcion sirve para leer rootfiles y convertirlas en
    pandas.DataFrames --> 
       Ref1: https://www.geeksforgeeks.org/python-pandas-dataframe/
       Ref2: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    Esta funcion devuelve un pandas dataframe en el que cada fila
    es un suceso (un event) y cada columna, una variable 
    (e.g. lep1_pt seria una columna)
    '''
    # -- Abrimos la rootfile, y 
    #    convertimos el tree EVENTS en un array
    tfile = r.TFile.Open(file_ + ".root")    
    
    ttree = tfile.Get(treename)
    arr = tree2array(ttree, start = start, stop = stop)

    # -- Convertimos a dataframe
    df_out = pd.DataFrame(arr, columns=vars_)       

    return df_out

def add_friends(df, file_, vars_ = [], start = 0, stop = 100):
   ''' 
   Esta funcion esencialmente llama a load_data, pero para
   variables definidas en friend trees y incluye nuevas 
   columnas en el dataframe principal.
   ''' 
   file_ += "_Friend"
   
   df_friend = load_data(file_, vars_, treename = "Friends", start = start, stop = stop)
   df_out = combine_dataframes([df, df_friend], axis = 1)
   return df_out 

def label_dataframe(df, label):
    '''
    Funcion para etiquetar dataframes. Esto es necesario
    para poder hacer entrenamientos supervisados. 
    '''
    nentries = df.shape[0]
    etiqueta = np.array( [label]*nentries )
    df_label = pd.DataFrame(etiqueta, columns = ["is_signal"])
    df = combine_dataframes([df, df_label], axis = 1)
    return df
    
