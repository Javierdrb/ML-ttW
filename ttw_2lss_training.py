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
from tensorflow import keras

if __name__ == "__main__":
    mainpath = "/beegfs/data/TOPnanoAODv6/ttW_MC_Ntuples_skim/2016/"
    muestras = {"ttbar"      : ["TTLep_pow_part1"], 
                "ttW"        : ["TTWToLNu_PSW"]}
    # -- Variables que vamos a usar para procesar datos:
    vars_keep = ["nLepGood"]
    vars_friend_recl = ["nJet25_Recl"] 

    # -- Muestras que vamos a utilizar
    ttbar = muestras["ttbar"][0]
    ttw = muestras["ttW"][0]
    
    # -- Nos cargamos los dataframes con la funcion load_data (definida en read_data.py)
    df_ttw   = load_data(mainpath + ttw, vars_keep)
    df_ttw = label_dataframe(df_ttw, 1)
    print(df_ttw)
    df_ttw   = add_friends(df_ttw, mainpath+"1_recl_enero/"+ttw, vars_friend_recl)
    print(df_ttw)

    df_ttbar = load_data(mainpath + ttbar, vars_keep)
    df_ttbar = add_friends(df_ttbar, mainpath+"1_recl_enero/"+ttbar, vars_friend_recl)
    df_ttbar = label_dataframe(df_ttbar, 0)

   
    # Combinamos todos los dataframes
    dfs_to_combine = [df_ttw, df_ttbar]
    df = combine_dataframes(dfs_to_combine, axis = 0) # Para concatenar filas
    print(df)

    vars_train = ["nLepGood"]
    X_train, X_test, y_train, y_test = train_test_split(df[vars_train], df['is_signal'], test_size=0.5, random_state=1)
    df_train=pd.concat([X_train,y_train], axis=1)
    df_validation=pd.concat([X_test,y_test], axis=1)



    model = keras.models.Sequential()
    normal_ini=keras.initializers.glorot_normal(seed=None)
    model.add(keras.layers.Dense(64,  input_shape = (len(vars_train),), activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.summary()
    
    sgd = keras.optimizers.SGD(lr=0.01)
    adamax=keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adamax, metrics = ["accuracy"])
    histObj = model.fit(df_train[vars_train], keras.utils.to_categorical(df_train["is_signal"]), epochs=5, batch_size=1000,validation_data=(df_validation[vars_train], keras.utils.to_categorical(df_validation["is_signal"])))
    
    
    #Prediccion sobre test
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    
    print("Running on test sample. This may take a moment.")
    probs = model.predict(df_validation[vars_train])#predict probability over test sample
    AUC = roc_auc_score(df_validation["is_signal"], probs[:,1])
    print("Test Area under Curve = {0}".format(AUC))
    
    #Prediccion sobre train
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    
    print("Running on test sample. This may take a moment.")
    probs2 = model.predict(df_train[vars_train])#predict probability over test sample
    AUC2 = roc_auc_score(df_train["is_signal"], probs2[:,1])
    print("Test Area under Curve = {0}".format(AUC2))
