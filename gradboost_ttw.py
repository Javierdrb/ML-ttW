''' Implementation of a neural network to discriminate ttW ''' #just trying
# -- Import libraries 
import pandas as pd
import numpy as np
import ROOT as r
import df_utils as dfu
from samples import smp2df
import matplotlib.pyplot as plt
from functions import *
from grid import *

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
import xgboost as xgb

import time
start_time = time.time()

#pd.set_option('display.max_rows',None)
#pd.set_option('display.max_columns',None)


def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))


if __name__ == "__main__":
	
###########################					BUILDING THE DATAFRAME												##############################	
	
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
    
    
    
    
    # Create the signal dataframe
    smp_ttw = smp2df(branches = branches,  #sample2df comvierte sample a data frame
                     friends = friends, 
                     name = "df_ttw")
    
    
            
    for year in [2016, 2017, 2018]:
      smp_ttw.load_data(                   #loaddata en samples.py
        path = mainpath + "%s"%year, 
        selection = "nLepGood==2",  #&(JetSel_Recl_btagDeepFlavB[0]>=0.5)",    #Selection to add in case we deal with certain requirement (for instance if we add a variable on jets, we must ask for having at least one jet)
        #stop=1000000,
        stop=21000,
        #stop=50,
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
        #stop=1000000,
        stop=80000,
        process = samples["ttbar"][year]) 
    smp_tt.label_dataframe(val = 0)   #check that since this is bckg, val must be 0
    
    df_ttw = smp_ttw.df
    df_ttbar = smp_tt.df

    print(df_ttw)
    
   
    
  
    # Combinamos todos los dataframes
    dfs_to_combine = [df_ttw, df_ttbar]
    df = dfu.combine_dataframes(dfs_to_combine, axis = 0) # Para concatenar filas
    
    
    
  
  
    ###############################      COMPLEX VARIABLE DEFINITION TO ADD TO THE 'SIMPLE' DATAFRAME  					############################  
    
    ######		B-tagging business #####
    df=df.assign(B1_pt=100)
    df=df.assign(B1_eta=100)
    df=df.assign(B1_phi=100)
    df=df.assign(B1_mass=100)
    df=df.assign(B2_pt=100)
    df=df.assign(B2_eta=100)
    df=df.assign(B2_phi=100)
    df=df.assign(B2_mass=100)
    
    
    btagging(df,workingpoint="Tight")
    
    
    
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
    
    #df["deltaphilj11"]=deltaphi(l1,j1)
    #df["deltaphilj12"]=deltaphi(l1,j2)
    #df["deltaphilj21"]=deltaphi(l2,j1)
    #df["deltaphilj22"]=deltaphi(l2,j2)
    
    #df["deltarlj11"]=deltar(l1,j1)
    #df["deltarlj12"]=deltar(l1,j2)
    #df["deltarlj21"]=deltar(l2,j1)
    #df["deltarlj22"]=deltar(l2,j2)
    
    df["combipt"]=combipt(l1,l2)
    #df["deltaetalep"]=deltaeta(l1,l2)
    #df["deltaphilep"]=deltaphi(l1,l2)
    #df["deltarlep"]=deltar(l1,l2)
    df["deltarjet"]=deltar(j1,j2)     #dejar para xgb
    
    
    
    #df["deltarlb11"]=deltar(l1,b1)
    #df["deltarlb12"]=deltar(l1,b2)
    #df["deltarlb21"]=deltar(l2,b1)
    #df["deltarlb22"]=deltar(l2,b2)
    
    df["deltarjb11"]=deltar(j1,b1)    #dejar para xgb
    #df["deltarjb12"]=deltar(j1,b2)
    #df["deltarjb21"]=deltar(j2,b1)
    #df["deltarjb22"]=deltar(j2,b2)
    
    df["notBjets"]=df["nJet25_Recl"]-df["nBJetLoose25_Recl"]
    
    
    
    flavouring(df)
    charge(df)
    print(df)
    
    
    
    
    
 ########################################				STARTING OF THE MACHINE LEARNING MODELS						#############################   
    
      
    #Variables to add to the training. We can split a diferent selection of variables for each model:
   
    vars_train_gboost=['htJet25j_Recl','jet3_pt' ,'jet2_pt' ,'year', 'jet4_pt', 'combipt', 'lep1_charge', 'lep2_mass', 'mlj12','lep2_elec',
     'lep2_pt', 'Flav_muon','B2_pt' ,'mlj11' ,'Flav_elec' ,'nBJetMedium25_Recl', 'deltarjb11' ,'jet3_eta','deltarjet', 'jet5_pt'] 
     
     
     #		Splitting of datasets (for each of the training vars subset)
     
    X_train_gboost, X_test_gboost, y_train_gboost, y_test_gboost = train_test_split(df[vars_train_gboost], df['is_signal'], test_size=0.3, random_state=1)
    df_train_gboost=pd.concat([X_train_gboost,y_train_gboost], axis=1)
    df_test_gboost=pd.concat([X_test_gboost,y_test_gboost], axis=1)
    
    X_test_gboost, X_validation_gboost, y_test_gboost, y_validation_gboost = train_test_split(df_test_gboost[vars_train_gboost], df_test_gboost['is_signal'], test_size=0.8, random_state=5)
    df_test_gboost=pd.concat([X_test_gboost,y_test_gboost], axis=1)
    df_validation_gboost=pd.concat([X_validation_gboost,y_validation_gboost], axis=1)
    
    bucle(X_train_gboost,y_train_gboost,X_test_gboost,y_test_gboost,df_test_gboost,df_train_gboost,vars_train_gboost)
    
    #gboost = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, n_jobs=-1,use_label_encoder=False)#ensemble.GradientBoostingClassifier(n_jobs=-1,max_depth=15, min_samples_leaf=5)				#Descomentar si quiero comparar con gradientboosting (pero solo se puede correr con pocos datos, boosting no paraleliza bien)
    #gboost.fit(X_train_gboost,y_train_gboost)  #Training of the GradBoost
    
    #print("Gboost AUC (test) = {0}".format(roc_auc_score(df_test_gboost['is_signal'],gboost.predict_proba(df_test_gboost[vars_train_gboost])[:,1])))
    #print("Gboost AUC (train) = {0}".format(roc_auc_score(df_train_gboost['is_signal'],gboost.predict_proba(df_train_gboost[vars_train_gboost])[:,1])))
    
    exit()
    
    
    
    
    
    
    
	
	
    
    #bucle(X_train_gboost,y_train_gboost,X_test_gboost,y_test_gboost,df_test_gboost,df_train_gboost,vars_train_gboost)
    exit()
    
		
		#gboost=xgb.XGBClassifier(n_estimators=100,max_depth=3,learning_rate=i,n_jobs=-1,use_label_encoder=False)
		#gboost.fit(X_train_gboost,y_train_gboost)
		#print("Learning rate={0}".format(i))
		#print("Gboost AUC (test) = {0}".format(roc_auc_score(df_test_gboost['is_signal'],gboost.predict_proba(df_test_gboost[vars_train_gboost])[:,1])))
		#print("Gboost AUC (train) = {0}".format(roc_auc_score(df_train_gboost['is_signal'],gboost.predict_proba(df_train_gboost[vars_train_gboost])[:,1])))
		
		
		
    
    
    
    ###############################						PLOTTING AND PRINTING RESULTS						##################################
    
    #VARIABLE IMPORTANCE
    features_list = df_train_gboost.columns.values
    feature_importance = gboost.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1][:20]
    sorted_idx_full=np.argsort(feature_importance)
    plt.figure(figsize=(13,10))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx][::-1], align='center')
    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx][::-1])
    plt.xlabel('Importance')
    plt.title('Feature importances')
    plt.savefig('importancia_variables_Gboost.png')
    
    
    #AUCs
    print("Gboost AUC (test) = {0}".format(roc_auc_score(df_test_gboost['is_signal'],gboost.predict_proba(df_test_gboost[vars_train_gboost])[:,1])))
    print("Gboost AUC (train) = {0}".format(roc_auc_score(df_train_gboost['is_signal'],gboost.predict_proba(df_train_gboost[vars_train_gboost])[:,1])))
    
    
    ##Plotting the AUCs
    f, ax = plt.subplots(figsize=(7,8))   #Medida del AUC (indicador cuan bueno es el modelo)
    
    roc_auc_plot(y_test_gboost,gboost.predict_proba(X_test_gboost),label='Gboost test',l='-')						
    roc_auc_plot(y_train_gboost,gboost.predict_proba(X_train_gboost),label='Gboost train',l='-')
    
    ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', label='Random Classifier')    
    ax.legend(loc="lower right")    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('Receiver Operator Characteristic curves')
    f.savefig('auc_gboost.png') 

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time/60, "minutes")
