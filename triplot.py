''' Implementation of a neural network to discriminate ttW ''' #just trying
# -- Import libraries 
import pandas as pd
import numpy as np
import ROOT as r
import df_utils as dfu
from samples import smp2df
import matplotlib.pyplot as plt
from functions import *

# -- ML libraries 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, auc, precision_recall_curve
from tensorflow import keras
from sklearn import tree, ensemble

import time
start_time = time.time()

#pd.set_option('display.max_rows',10)
#pd.set_option('display.max_columns',None)



def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))


if __name__ == "__main__":
    mainpath = "/beegfs/data/nanoAODv9/ttW_data_forFlips/mva_vars_v2/"#"/beegfs/data/TOPnanoAODv6/ttW_MC_Ntuples_skim/"

    # Dictionary with sample names
    samples = {
      "ttbar" : { 2016 : "TTLep_pow_part1_Friend", #Se mpueden meter listas
                  2017 : "TTLep_pow_Friend",
                  2018 : "TTLep_pow_Friend"}, 
      "ttW"   : { 2016 : "TTWToLNu_PSW_Friend", 
                  2017 : "TTWToLNu_PSW_Friend",
                  2018 : "TTWToLNu_fxfx_Friend"}
    }
    # -- Variables to read from trees (can be defined in a friend tree) 
    friends = ["1_recl_enero"]   
    branches = ["year", 
                         "nLepGood", 
                         "lep1_pt", "lep1_eta","lep1_phi","lep1_mass","lep1_pdgId",
                         "lep2_pt", "lep2_eta","lep2_phi","lep2_mass","lep2_pdgId",
                        # "lep3_pt", "lep3_eta","lep3_phi","lep3_mass",			For the moment, we will just do the training with events of 2 leptons
                         "nJet25_Recl", 
                         "htJet25j_Recl", 
                         "MET_pt", 
                         "nBJetLoose25_Recl",
                         "nBJetMedium25_Recl",
                         "nBJetLoose40_Recl",
                         "nBJetMedium40_Recl",
                         "jet1_pt", "jet1_eta","jet1_phi","jet1_mass","jet1_btagDeepFlavB", 
                         "jet2_pt", "jet2_eta","jet2_phi","jet2_mass","jet2_btagDeepFlavB",
                         "jet3_pt", "jet3_eta","jet3_phi","jet3_mass","jet3_btagDeepFlavB",
                         "jet4_pt", "jet4_eta","jet4_phi","jet4_mass","jet4_btagDeepFlavB",
                         "jet5_pt", "jet5_eta","jet5_phi","jet5_mass","jet5_btagDeepFlavB",
                         "jet6_pt", "jet6_eta","jet6_phi","jet6_mass","jet6_btagDeepFlavB",
                         "jet7_pt", "jet7_eta","jet7_phi","jet7_mass","jet7_btagDeepFlavB"]
    
    
########						FIRST (LOOSE WP)								######  
    
    # Create the signal dataframe
    smp_ttw = smp2df(branches = branches,  #sample2df comvierte sample a data frame
                     friends = friends, 
                     name = "df_ttw")
    
    
            
    for year in [2016, 2017, 2018]:
      smp_ttw.load_data(                   #loaddata en samples.py
        path = mainpath + "%s"%year, 
        selection = "nLepGood==2",  #&(JetSel_Recl_btagDeepFlavB[0]>=0.5)",    #Selection to add in case we deal with certain requirement (for instance if we add a variable on jets, we must ask for having at least one jet)
        stop=1000000,
        process = samples["ttW"][year]) 
    smp_ttw.label_dataframe(val = 1)   #etiquetado (señal 1 bck 0)

    # Create the bkg dataframe
    smp_tt = smp2df(branches = branches, 
                    friends = friends, 
                    name = "df_tt")
    for year in [2016, 2017, 2018]:
      smp_tt.load_data(
        path = mainpath + "%s"%year, 
        selection = "nLepGood==2",#"nJet25_Recl<=5",
        stop=1000000,
        process = samples["ttbar"][year]) 
    smp_tt.label_dataframe(val = 0)   #check that since this is bckg, val must be 0
    
    df_ttw = smp_ttw.df
    df_ttbar = smp_tt.df

    print(df_ttw)
    
    ###############################Definicion variables 'complejas' 
    
    
    
    
    
    # Combinamos todos los dataframes
    dfs_to_combine = [df_ttw, df_ttbar]
    df = dfu.combine_dataframes(dfs_to_combine, axis = 0) # Para concatenar filas
    
    
    ######		B-tagging business #####
    df=df.assign(B1_pt=100)
    df=df.assign(B1_eta=100)
    df=df.assign(B1_phi=100)
    df=df.assign(B1_mass=100)
    #df=df.assign(B2_pt=100)
    #df=df.assign(B2_eta=100)
    #df=df.assign(B2_phi=100)
    #df=df.assign(B2_mass=100)
    
    
    btagging(df,workingpoint="Loose")
    
    
    
    
    ############################### Complex variable definition
    
    # Create 4 vectors
    l1 = create4vec(df["lep1_pt"], df["lep1_eta"], df["lep1_phi"], df["lep1_mass"])
    l2 = create4vec(df["lep2_pt"], df["lep2_eta"], df["lep2_phi"], df["lep2_mass"])
    
    j1 = create4vec(df["jet1_pt"], df["jet1_eta"], df["jet1_phi"], df["jet1_mass"])
    j2 = create4vec(df["jet2_pt"], df["jet2_eta"], df["jet2_phi"], df["jet2_mass"])
    
    b1=create4vec(df["B1_pt"],df["B1_eta"],df["B1_phi"],df["B1_mass"])
    #b2=create4vec(df["B2_pt"],df["B2_eta"],df["B2_phi"],df["B2_mass"])
    
    
    
    df["mll"] = mll(l1, l2)
    
    df["mlj11"]=mll(l1,j1)
    df["mlj22"]=mll(l2,j2)
    df["mlj12"]=mll(l1,j2)
    df["mlj21"]=mll(l2,j1)
    
    df["deltaphilj11"]=deltaphi(l1,j1)
    df["deltaphilj12"]=deltaphi(l1,j2)
    df["deltaphilj21"]=deltaphi(l2,j1)
    df["deltaphilj22"]=deltaphi(l2,j2)
    
    df["deltarlj11"]=deltar(l1,j1)
    df["deltarlj12"]=deltar(l1,j2)
    df["deltarlj21"]=deltar(l2,j1)
    df["deltarlj22"]=deltar(l2,j2)
    
    df["combipt"]=combipt(l1,l2)
    df["deltaetalep"]=deltaeta(l1,l2)
    df["deltaphilep"]=deltaphi(l1,l2)
    df["deltarlep"]=deltar(l1,l2)
    df["deltarjet"]=deltar(j1,j2)
    
    
    
    df["deltarlb11"]=deltar(l1,b1)
    #df["deltarlb12"]=deltar(l1,b2)
    df["deltarlb21"]=deltar(l2,b1)
    #df["deltarlb22"]=deltar(l2,b2)
    
    df["deltarjb11"]=deltar(j1,b1)
    #df["deltarjb12"]=deltar(j1,b2)
    df["deltarjb21"]=deltar(j2,b1)
    #df["deltarjb22"]=deltar(j2,b2)
    
    df["notBjets"]=df["nJet25_Recl"]-df["nBJetLoose25_Recl"]
    
    
    
    flavouring(df)
    charge(df)
    print(df)
    
    
    
    
    
    
    #Variables to add to the training
    #vars_train = ["lep1_pt","lep2_pt","jet1_pt","MET_pt","mll","deltarjet","htJet25j_Recl","deltarlep",
    # "jet1_btagDeepFlavB","jet2_btagDeepFlavB","deltarlj","deltarblep","deltarbj","combipt"]
    vars_train=["notBjets","deltaetalep","deltaphilep","deltarjet","deltarlep","combipt","mll",
                         "deltaphilj11","deltaphilj12","deltaphilj21","deltaphilj22",
                         "deltarlj11","deltarlj21","deltarlj12","deltarlj22",
                         "deltarlb11","deltarlb21",#"deltarlb12","deltarlb22",
                         "deltarjb11","deltarjb21",#"deltarjb12","deltarjb22",
                         "mlj11","mlj12","mlj21","mlj22",
                         "Flav_elec","Flav_muon","Flav_mix",
                         "B1_pt","B1_eta","B1_phi","B1_mass",
                         #"B2_pt","B2_eta","B2_phi","B2_mass",
                         "year", 
                         "nLepGood", 
                         "lep1_pt", "lep1_eta","lep1_phi","lep1_mass","lep1_charge","lep1_elec",
                         "lep2_pt", "lep2_eta","lep2_phi","lep2_mass","lep2_elec","lep2_charge",
                         "nJet25_Recl", 
                         "htJet25j_Recl", 
                         "MET_pt", 
                         "nBJetLoose25_Recl",
                         "nBJetMedium25_Recl",
                         "nBJetLoose40_Recl",
                         "nBJetMedium40_Recl",
                         "jet1_pt", "jet1_eta","jet1_phi","jet1_mass","jet1_btagDeepFlavB", 
                         "jet2_pt", "jet2_eta","jet2_phi","jet2_mass","jet2_btagDeepFlavB",
                         "jet3_pt", "jet3_eta","jet3_phi","jet3_mass","jet3_btagDeepFlavB",
                         "jet4_pt", "jet4_eta","jet4_phi","jet4_mass","jet4_btagDeepFlavB",
                         "jet5_pt", "jet5_eta","jet5_phi","jet5_mass","jet5_btagDeepFlavB",
                         "jet6_pt", "jet6_eta","jet6_phi","jet6_mass","jet6_btagDeepFlavB",
                         "jet7_pt", "jet7_eta","jet7_phi","jet7_mass","jet7_btagDeepFlavB"]
    
    
    # From Andrea
    X_train, X_test, y_train, y_test = train_test_split(df[vars_train], df['is_signal'], test_size=0.3, random_state=1)
    df_train=pd.concat([X_train,y_train], axis=1)
    df_test=pd.concat([X_test,y_test], axis=1)
    
    X_test, X_validation, y_test, y_validation = train_test_split(df_test[vars_train], df_test['is_signal'], test_size=0.8, random_state=5)
    df_test=pd.concat([X_test,y_test], axis=1)
    df_validation=pd.concat([X_validation,y_validation], axis=1)
    
    
    RF=RandomForestClassifier(n_jobs=-1,min_samples_leaf=1000,max_depth=50,min_samples_split=2)
    RF.fit(X_train,y_train)  #Trainig of the RF
    
    
    
    
    
    
    
    model = keras.models.Sequential()
    normal_ini=keras.initializers.glorot_normal(seed=None)
    model.add(keras.layers.Dense(64,  input_shape = (len(vars_train),), activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.summary()
    
    
    
    
    sgd = keras.optimizers.SGD(lr=0.01)
    #adamax=keras.optimizers.Adamax(learning_rate=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    adam=keras.optimizers.Adam(lr=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics = ["accuracy"])
    histObj = model.fit(df_train[vars_train], keras.utils.to_categorical(df_train["is_signal"]), epochs=35, batch_size=1000,shuffle=True,validation_data=(df_validation[vars_train], keras.utils.to_categorical(df_validation["is_signal"])))

    
    ####Prediccion sobre test
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    print("Running on test sample. This may take a moment.")
    probs = model.predict(df_test[vars_train])#predict probability over test sample
    AUC = roc_auc_score(df_test["is_signal"], probs[:,1])
    print("Test Area under Curve = {0}".format(AUC))
    
    ######Prediccion sobre train
    print("Running on train sample. This may take a moment.")
    probs2 = model.predict(df_train[vars_train])#predict probability over train sample
    AUC2 = roc_auc_score(df_train["is_signal"], probs2[:,1])
    print("Train Area under Curve = {0}".format(AUC2))
    
    #plotLearningCurves(histObj)
    
###### 									END OF LOOSE WP 				######


#####									MEDIUM WP    					########
    
    # Create the signal dataframe
    smp_ttw1 = smp2df(branches = branches,  #sample2df comvierte sample a data frame
                     friends = friends, 
                     name = "df_ttw1")
    
    
            
    for year in [2016, 2017, 2018]:
      smp_ttw1.load_data(                   #loaddata en samples.py
        path = mainpath + "%s"%year, 
        selection = "nLepGood==2",  #&(JetSel_Recl_btagDeepFlavB[0]>=0.5)",    #Selection to add in case we deal with certain requirement (for instance if we add a variable on jets, we must ask for having at least one jet)
        stop=1000000,
        process = samples["ttW"][year]) 
    smp_ttw1.label_dataframe(val = 1)   #etiquetado (señal 1 bck 0)

    # Create the bkg dataframe
    smp_tt1 = smp2df(branches = branches, 
                    friends = friends, 
                    name = "df_tt1")
    for year in [2016, 2017, 2018]:
      smp_tt1.load_data(
        path = mainpath + "%s"%year, 
        selection = "nLepGood==2",#"nJet25_Recl<=5",
        stop=1000000,
        process = samples["ttbar"][year]) 
    smp_tt1.label_dataframe(val = 0)   #check that since this is bckg, val must be 0
    
    df_ttw1 = smp_ttw1.df
    df_ttbar1 = smp_tt1.df

    print(df_ttw1)
    
    ###############################Definicion variables 'complejas' 
    
    
    
    
    
    # Combinamos todos los dataframes
    dfs_to_combine1 = [df_ttw1, df_ttbar1]
    df1 = dfu.combine_dataframes(dfs_to_combine1, axis = 0) # Para concatenar filas
    
    
    ######		B-tagging business #####
    df1=df1.assign(B1_pt=100)
    df1=df1.assign(B1_eta=100)
    df1=df1.assign(B1_phi=100)
    df1=df1.assign(B1_mass=100)
    #df1=df1.assign(B2_pt=100)
    #df1=df1.assign(B2_eta=100)
    #df1=df1.assign(B2_phi=100)
    #df1=df1.assign(B2_mass=100)
    
    
    btagging(df1,workingpoint="Medium")
    
    
    
    
    ############################### Complex variable definition
    
    # Create 4 vectors
    l1_1 = create4vec(df1["lep1_pt"], df1["lep1_eta"], df1["lep1_phi"], df1["lep1_mass"])
    l2_1 = create4vec(df1["lep2_pt"], df1["lep2_eta"], df1["lep2_phi"], df1["lep2_mass"])
    
    j1_1 = create4vec(df1["jet1_pt"], df1["jet1_eta"], df1["jet1_phi"], df1["jet1_mass"])
    j2_1 = create4vec(df1["jet2_pt"], df1["jet2_eta"], df1["jet2_phi"], df1["jet2_mass"])
    
    b1_1=create4vec(df1["B1_pt"],df1["B1_eta"],df1["B1_phi"],df1["B1_mass"])
    #b2_1=create4vec(df1["B2_pt"],df1["B2_eta"],df1["B2_phi"],df1["B2_mass"])
    
    
    
    df1["mll"] = mll(l1_1, l2_1)
    
    df1["mlj11"]=mll(l1_1,j1_1)
    df1["mlj22"]=mll(l2_1,j2_1)
    df1["mlj12"]=mll(l1_1,j2_1)
    df1["mlj21"]=mll(l2_1,j1_1)
    
    df1["deltaphilj11"]=deltaphi(l1_1,j1_1)
    df1["deltaphilj12"]=deltaphi(l1_1,j2_1)
    df1["deltaphilj21"]=deltaphi(l2_1,j1_1)
    df1["deltaphilj22"]=deltaphi(l2_1,j2_1)
    
    df1["deltarlj11"]=deltar(l1_1,j1_1)
    df1["deltarlj12"]=deltar(l1_1,j2_1)
    df1["deltarlj21"]=deltar(l2_1,j1_1)
    df1["deltarlj22"]=deltar(l2_1,j2_1)
    
    df1["combipt"]=combipt(l1_1,l2_1)
    df1["deltaetalep"]=deltaeta(l1_1,l2_1)
    df1["deltaphilep"]=deltaphi(l1_1,l2_1)
    df1["deltarlep"]=deltar(l1_1,l2_1)
    df1["deltarjet"]=deltar(j1_1,j2_1)
    
    
    
    df1["deltarlb11"]=deltar(l1_1,b1_1)
    #df1["deltarlb12"]=deltar(l1_1,b2_1)
    df1["deltarlb21"]=deltar(l2_1,b1_1)
    #df1["deltarlb22"]=deltar(l2_1,b2_1)
    
    df1["deltarjb11"]=deltar(j1_1,b1_1)
    #df1["deltarjb12"]=deltar(j1_1,b2_1)
    df1["deltarjb21"]=deltar(j2_1,b1_1)
    #df1["deltarjb22"]=deltar(j2_1,b2_1)
    
    df1["notBjets"]=df1["nJet25_Recl"]-df1["nBJetLoose25_Recl"]
    
    
    
    flavouring(df1)
    charge(df1)
    print(df1)
    
    
    
    
    
    
    #Variables to add to the training
    #vars_train = ["lep1_pt","lep2_pt","jet1_pt","MET_pt","mll","deltarjet","htJet25j_Recl","deltarlep",
    # "jet1_btagDeepFlavB","jet2_btagDeepFlavB","deltarlj","deltarblep","deltarbj","combipt"]
    vars_train=["notBjets","deltaetalep","deltaphilep","deltarjet","deltarlep","combipt","mll",
                         "deltaphilj11","deltaphilj12","deltaphilj21","deltaphilj22",
                         "deltarlj11","deltarlj12","deltarlj21","deltarlj22",
                         "deltarlb11","deltarlb21",#"deltarlb12","deltarlb22",
                         "deltarjb11","deltarjb21",#"deltarjb12","deltarjb22",
                         "mlj11","mlj12","mlj21","mlj22",
                         "Flav_elec","Flav_muon","Flav_mix",
                         "B1_pt","B1_eta","B1_phi","B1_mass",
                         #"B2_pt","B2_eta","B2_phi","B2_mass",
                         "year", 
                         "nLepGood", 
                         "lep1_pt", "lep1_eta","lep1_phi","lep1_mass","lep1_charge","lep1_elec",
                         "lep2_pt", "lep2_eta","lep2_phi","lep2_mass","lep2_elec","lep2_charge",
                         "nJet25_Recl", 
                         "htJet25j_Recl", 
                         "MET_pt", 
                         "nBJetLoose25_Recl",
                         "nBJetMedium25_Recl",
                         "nBJetLoose40_Recl",
                         "nBJetMedium40_Recl",
                         "jet1_pt", "jet1_eta","jet1_phi","jet1_mass","jet1_btagDeepFlavB", 
                         "jet2_pt", "jet2_eta","jet2_phi","jet2_mass","jet2_btagDeepFlavB",
                         "jet3_pt", "jet3_eta","jet3_phi","jet3_mass","jet3_btagDeepFlavB",
                         "jet4_pt", "jet4_eta","jet4_phi","jet4_mass","jet4_btagDeepFlavB",
                         "jet5_pt", "jet5_eta","jet5_phi","jet5_mass","jet5_btagDeepFlavB",
                         "jet6_pt", "jet6_eta","jet6_phi","jet6_mass","jet6_btagDeepFlavB",
                         "jet7_pt", "jet7_eta","jet7_phi","jet7_mass","jet7_btagDeepFlavB"]
    
    
    # From Andrea
    X_train1, X_test1, y_train1, y_test1 = train_test_split(df1[vars_train], df1['is_signal'], test_size=0.3, random_state=1)
    df_train1=pd.concat([X_train1,y_train1], axis=1)
    df_test1=pd.concat([X_test1,y_test1], axis=1)
    
    X_test1, X_validation1, y_test1, y_validation1 = train_test_split(df_test1[vars_train], df_test1['is_signal'], test_size=0.8, random_state=5)
    df_test1=pd.concat([X_test1,y_test1], axis=1)
    df_validation1=pd.concat([X_validation1,y_validation1], axis=1)
    
    
    RF1=RandomForestClassifier(n_jobs=-1,min_samples_leaf=1000,max_depth=50,min_samples_split=2)
    RF1.fit(X_train1,y_train1)  #Trainig of the RF
    
    
    
    
    
    
    
    model1 = keras.models.Sequential()
    normal_ini=keras.initializers.glorot_normal(seed=None)
    model1.add(keras.layers.Dense(64,  input_shape = (len(vars_train),), activation='relu'))
    model1.add(keras.layers.Dense(32, activation='relu'))
    model1.add(keras.layers.Dense(16, activation='relu'))
    model1.add(keras.layers.Dense(16, activation='relu'))
    model1.add(keras.layers.Dense(8, activation='relu'))
    model1.add(keras.layers.Dense(2, activation='softmax'))
    model1.summary()
    
    
    
    
    sgd = keras.optimizers.SGD(lr=0.01)
    #adamax=keras.optimizers.Adamax(learning_rate=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    adam=keras.optimizers.Adam(lr=0.0005)
    model1.compile(loss='binary_crossentropy', optimizer=adam, metrics = ["accuracy"])
    histObj1 = model1.fit(df_train1[vars_train], keras.utils.to_categorical(df_train1["is_signal"]), epochs=35, batch_size=1000,shuffle=True,validation_data=(df_validation1[vars_train], keras.utils.to_categorical(df_validation1["is_signal"])))

    
    ####Prediccion sobre test
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    print("Running on test sample. This may take a moment.")
    probs1 = model1.predict(df_test1[vars_train])#predict probability over test sample
    AUC1 = roc_auc_score(df_test1["is_signal"], probs1[:,1])
    print("Test Area under Curve = {0}".format(AUC1))
    
    ######Prediccion sobre train
    print("Running on train sample. This may take a moment.")
    probs21 = model1.predict(df_train1[vars_train])#predict probability over train sample
    AUC21 = roc_auc_score(df_train1["is_signal"], probs21[:,1])
    print("Train Area under Curve = {0}".format(AUC21))
    
####################					END OF MEDIUM WP			#################3

####################					START OF TIGHT WP           #####################
    smp_ttw2 = smp2df(branches = branches,  #sample2df comvierte sample a data frame
                     friends = friends, 
                     name = "df_ttw2")
    
    
            
    for year in [2016, 2017, 2018]:
      smp_ttw2.load_data(                   #loaddata en samples.py
        path = mainpath + "%s"%year, 
        selection = "nLepGood==2",  #&(JetSel_Recl_btagDeepFlavB[0]>=0.5)",    #Selection to add in case we deal with certain requirement (for instance if we add a variable on jets, we must ask for having at least one jet)
        stop=1000000,
        process = samples["ttW"][year]) 
    smp_ttw2.label_dataframe(val = 1)   #etiquetado (señal 1 bck 0)

    # Create the bkg dataframe
    smp_tt2 = smp2df(branches = branches, 
                    friends = friends, 
                    name = "df_tt2")
    for year in [2016, 2017, 2018]:
      smp_tt2.load_data(
        path = mainpath + "%s"%year, 
        selection = "nLepGood==2",#"nJet25_Recl<=5",
        stop=1000000,
        process = samples["ttbar"][year]) 
    smp_tt2.label_dataframe(val = 0)   #check that since this is bckg, val must be 0
    
    df_ttw2 = smp_ttw2.df
    df_ttbar2 = smp_tt2.df

    print(df_ttw2)
    
    ###############################Definicion variables 'complejas' 
    
    
    
    
    
    # Combinamos todos los dataframes
    dfs_to_combine2 = [df_ttw2, df_ttbar2]
    df2 = dfu.combine_dataframes(dfs_to_combine2, axis = 0) # Para concatenar filas
    
    
    ######		B-tagging business #####
    df2=df2.assign(B1_pt=100)
    df2=df2.assign(B1_eta=100)
    df2=df2.assign(B1_phi=100)
    df2=df2.assign(B1_mass=100)
    #df2=df2.assign(B2_pt=100)
    #df2=df2.assign(B2_eta=100)
    #df2=df2.assign(B2_phi=100)
    #df2=df2.assign(B2_mass=100)
    
    
    btagging(df2,workingpoint="Tight")
    
    
    
    
    ############################### Complex variable definition
    
    # Create 4 vectors
    l1_2 = create4vec(df2["lep1_pt"], df2["lep1_eta"], df2["lep1_phi"], df2["lep1_mass"])
    l2_2 = create4vec(df2["lep2_pt"], df2["lep2_eta"], df2["lep2_phi"], df2["lep2_mass"])
    
    j1_2 = create4vec(df2["jet1_pt"], df2["jet1_eta"], df2["jet1_phi"], df2["jet1_mass"])
    j2_2 = create4vec(df2["jet2_pt"], df2["jet2_eta"], df2["jet2_phi"], df2["jet2_mass"])
    
    b1_2=create4vec(df2["B1_pt"],df2["B1_eta"],df2["B1_phi"],df2["B1_mass"])
    #b2_2=create4vec(df2["B2_pt"],df2["B2_eta"],df2["B2_phi"],df2["B2_mass"])
    
    
    
    df2["mll"] = mll(l1_2, l2_2)
    
    df2["mlj11"]=mll(l1_2,j1_2)
    df2["mlj22"]=mll(l2_2,j2_2)
    df2["mlj12"]=mll(l1_2,j2_2)
    df2["mlj21"]=mll(l2_2,j1_2)
    
    df2["deltaphilj11"]=deltaphi(l1_2,j1_2)
    df2["deltaphilj12"]=deltaphi(l1_2,j2_2)
    df2["deltaphilj21"]=deltaphi(l2_2,j1_2)
    df2["deltaphilj22"]=deltaphi(l2_2,j2_2)
    
    df2["deltarlj11"]=deltar(l1_2,j1_2)
    df2["deltarlj12"]=deltar(l1_2,j2_2)
    df2["deltarlj21"]=deltar(l2_2,j1_2)
    df2["deltarlj22"]=deltar(l2_2,j2_2)
    
    df2["combipt"]=combipt(l1_2,l2_2)
    df2["deltaetalep"]=deltaeta(l1_2,l2_2)
    df2["deltaphilep"]=deltaphi(l1_2,l2_2)
    df2["deltarlep"]=deltar(l1_2,l2_2)
    df2["deltarjet"]=deltar(j1_2,j2_2)
    
    
    
    df2["deltarlb11"]=deltar(l1_2,b1_2)
   # df2["deltarlb12"]=deltar(l1_2,b2_2)
    df2["deltarlb21"]=deltar(l2_2,b1_2)
    #df2["deltarlb22"]=deltar(l2_2,b2_2)
    
    df2["deltarjb11"]=deltar(j1_2,b1_2)
   # df2["deltarjb12"]=deltar(j1_2,b2_2)
    df2["deltarjb21"]=deltar(j2_2,b1_2)
    #df2["deltarjb22"]=deltar(j2_2,b2_2)
    
    df2["notBjets"]=df2["nJet25_Recl"]-df2["nBJetLoose25_Recl"]
    
    
    
    flavouring(df2)
    charge(df2)
    print(df2)
    
    
    
    
    
    
    #Variables to add to the training
    #vars_train = ["lep1_pt","lep2_pt","jet1_pt","MET_pt","mll","deltarjet","htJet25j_Recl","deltarlep",
    # "jet1_btagDeepFlavB","jet2_btagDeepFlavB","deltarlj","deltarblep","deltarbj","combipt"]
    vars_train=["notBjets","deltaetalep","deltaphilep","deltarjet","deltarlep","combipt","mll",
                         "deltaphilj11","deltaphilj12","deltaphilj21","deltaphilj22",
                         "deltarlj11","deltarlj12","deltarlj21","deltarlj22",
                         "deltarlb11","deltarlb21",#"deltarlb12","deltarlb22",
                         "deltarjb11","deltarjb21",#"deltarjb12","deltarjb22",
                         "mlj11","mlj12","mlj21","mlj22",
                         "Flav_elec","Flav_muon","Flav_mix",
                         "B1_pt","B1_eta","B1_phi","B1_mass",
                         #"B2_pt","B2_eta","B2_phi","B2_mass",
                         "year", 
                         "nLepGood", 
                         "lep1_pt", "lep1_eta","lep1_phi","lep1_mass","lep1_charge","lep1_elec",
                         "lep2_pt", "lep2_eta","lep2_phi","lep2_mass","lep2_elec","lep2_charge",
                         "nJet25_Recl", 
                         "htJet25j_Recl", 
                         "MET_pt", 
                         "nBJetLoose25_Recl",
                         "nBJetMedium25_Recl",
                         "nBJetLoose40_Recl",
                         "nBJetMedium40_Recl",
                         "jet1_pt", "jet1_eta","jet1_phi","jet1_mass","jet1_btagDeepFlavB", 
                         "jet2_pt", "jet2_eta","jet2_phi","jet2_mass","jet2_btagDeepFlavB",
                         "jet3_pt", "jet3_eta","jet3_phi","jet3_mass","jet3_btagDeepFlavB",
                         "jet4_pt", "jet4_eta","jet4_phi","jet4_mass","jet4_btagDeepFlavB",
                         "jet5_pt", "jet5_eta","jet5_phi","jet5_mass","jet5_btagDeepFlavB",
                         "jet6_pt", "jet6_eta","jet6_phi","jet6_mass","jet6_btagDeepFlavB",
                         "jet7_pt", "jet7_eta","jet7_phi","jet7_mass","jet7_btagDeepFlavB"]
    
    
    # From Andrea
    X_train2, X_test2, y_train2, y_test2 = train_test_split(df2[vars_train], df2['is_signal'], test_size=0.3, random_state=1)
    df_train2=pd.concat([X_train2,y_train2], axis=1)
    df_test2=pd.concat([X_test2,y_test2], axis=1)
    
    X_test2, X_validation2, y_test2, y_validation2 = train_test_split(df_test2[vars_train], df_test2['is_signal'], test_size=0.8, random_state=5)
    df_test2=pd.concat([X_test2,y_test2], axis=1)
    df_validation2=pd.concat([X_validation2,y_validation2], axis=1)
    
    
    RF2=RandomForestClassifier(n_jobs=-1,min_samples_leaf=1000,max_depth=50,min_samples_split=2)
    RF2.fit(X_train2,y_train2)  #Trainig of the RF
    
    
    
    
    
    
    
    model2 = keras.models.Sequential()
    normal_ini=keras.initializers.glorot_normal(seed=None)
    model2.add(keras.layers.Dense(64,  input_shape = (len(vars_train),), activation='relu'))
    model2.add(keras.layers.Dense(32, activation='relu'))
    model2.add(keras.layers.Dense(16, activation='relu'))
    model2.add(keras.layers.Dense(16, activation='relu'))
    model2.add(keras.layers.Dense(8, activation='relu'))
    model2.add(keras.layers.Dense(2, activation='softmax'))
    model2.summary()
    
    
    
    
    sgd = keras.optimizers.SGD(lr=0.01)
    #adamax=keras.optimizers.Adamax(learning_rate=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    adam=keras.optimizers.Adam(lr=0.0005)
    model2.compile(loss='binary_crossentropy', optimizer=adam, metrics = ["accuracy"])
    histObj2 = model2.fit(df_train2[vars_train], keras.utils.to_categorical(df_train2["is_signal"]), epochs=35, batch_size=1000,shuffle=True,validation_data=(df_validation2[vars_train], keras.utils.to_categorical(df_validation2["is_signal"])))

    
    ####Prediccion sobre test
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    print("Running on test sample. This may take a moment.")
    probs2 = model2.predict(df_test2[vars_train])#predict probability over test sample
    AUC2 = roc_auc_score(df_test2["is_signal"], probs2[:,1])
    print("Test Area under Curve = {0}".format(AUC2))
    
    ######Prediccion sobre train
    print("Running on train sample. This may take a moment.")
    probs22 = model2.predict(df_train2[vars_train])#predict probability over train sample
    AUC22 = roc_auc_score(df_train2["is_signal"], probs22[:,1])
    print("Train Area under Curve = {0}".format(AUC22))
    
#####################			END OF TIGHT WP					####################
    
    
    
    f, ax = plt.subplots(figsize=(10,10))   #Medida del AUC (indicador cuan bueno es el modelo) (6,6 default)
    roc_auc_plot(y_test,RF.predict_proba(X_test),label='Loose WP (FOREST) ',l='--')
    roc_auc_plot(y_test1,RF1.predict_proba(X_test1),label='Medium WP (FOREST) ')
    roc_auc_plot(y_test2,RF2.predict_proba(X_test2),label='Tight WP (FOREST) ')
   # roc_auc_plot(y_train,RF.predict_proba(X_train),label='Forest train')
    #roc_auc_plot(y_test,gboost.predict_proba(X_test),label='GBOOST',l='-')						#Descomentar si gboost
    roc_auc_plot(y_test,model.predict(df_test[vars_train]),label='Loose WP (NN)',l='-.')
    roc_auc_plot(y_test1,model1.predict(df_test1[vars_train]),label='Medium WP (NN)')
    roc_auc_plot(y_test2,model2.predict(df_test2[vars_train]),label='Tight WP (NN)')
    #roc_auc_plot(y_train,model.predict(df_train[vars_train]),label='NN train')
    ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', label='Random Classifier')    
    ax.legend(loc="lower right")    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('Receiver Operator Characteristic curves')
    f.savefig('auc_tri.png')    
    
    print("Loose RF AUC={0}".format(roc_auc_score(df_test['is_signal'],RF.predict_proba(df_test[vars_train])[:,1])))
    print("Medium RF AUC={0}".format(roc_auc_score(df_test1['is_signal'],RF1.predict_proba(df_test1[vars_train])[:,1])))
    print("Tight RF AUC={0}".format(roc_auc_score(df_test2['is_signal'],RF2.predict_proba(df_test2[vars_train])[:,1])))

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")
