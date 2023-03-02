
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
import xgboost as xgb



#pd.set_option('display.max_rows',None)
#pd.set_option('display.max_columns',None)





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
        #stop=21000,
        stop=50,
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
        #stop=80000,
        process = samples["ttbar"][year]) 
    smp_tt.label_dataframe(val = 0)   #check that since this is bckg, val must be 0
    
    df_ttw = smp_ttw.df
    df_ttbar = smp_tt.df

    print(df_ttw)
    
    
    charge(df_ttbar)
    
    
    
    same_mask=(df_ttbar['lep2_charge']==df_ttbar['lep1_charge']).sum()
    diff_mask=(df_ttbar['lep2_charge']!=df_ttbar['lep1_charge']).sum()
    
    print('2lss: {0}\n 2los:{1}'.format(same_mask,diff_mask))
    print(df_ttbar.shape[0])
    
    exit()
