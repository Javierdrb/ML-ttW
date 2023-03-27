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
import xgboost as xgb

import time
start_time = time.time()

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))


if __name__ == "__main__":
	
###########################					BUILDING THE DATAFRAME												##############################	
	
    mainpath = '/nfs/fanae/user/jriego/TFM/CMSSW_10_4_0/src/CMGTools/TTHAnalysis/macros/carpeta_output_prueba/'
#"/beegfs/data/TOPnanoAODv6/ttW_MC_Ntuples_skim_mvaVars/mva_vars_v2/" los que estan con skim  #"/beegfs/data/TOPnanoAODv6/ttW_MC_Ntuples_skim/"

    # Dictionary with sample names
    samples = {
      #"ttbar" : { 2016 : "TTLep_pow_part1_Friend", #Se mpueden meter listas
       #           2017 : "TTLep_pow_Friend",
        #          2018 : "TTLep_pow_Friend"}, 
      #"ttW"   : { 2016 : "TTWToLNu_PSW_Friend", 
       #           2017 : "TTWToLNu_PSW_Friend",
        #          2018 : "TTWToLNu_fxfx_Friend"}
        "ttbar":{2016:"TTLep_pow_part1_Friend.chunk0"}
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
                         "jet7_pt", "jet7_eta","jet7_phi","jet7_mass","jet7_btagDeepFlavB",
                         "eventNumber"]
    
    
    
    
    # Create the signal dataframe
    #smp_ttw = smp2df(branches = branches,  #sample2df comvierte sample a data frame
     #                friends = friends, 
      #               name = "df_ttw")
    
    
            
    #for year in [2016, 2017, 2018]:
     # smp_ttw.load_data(                   #loaddata en samples.py
      #  path = mainpath + "%s"%year, 
       # selection = "nLepGood==2",  #&(JetSel_Recl_btagDeepFlavB[0]>=0.5)",    #Selection to add in case we deal with certain requirement (for instance if we add a variable on jets, we must ask for having at least one jet)
        #stop=1000000,
        #stop=21000,
        #stop=50,
        #process = samples["ttW"][year]) 
    #smp_ttw.label_dataframe(val = 1)   #etiquetado (señal 1 bck 0)

    # Create the bkg dataframe
    smp_tt = smp2df(branches = branches, 
                    friends = friends, 
                    name = "df_tt")
    for year in [2016]:#, 2017, 2018]:
      smp_tt.load_data(
        path = mainpath,# + "%s"%year, 
        #selection = "eventNumber%2!=0",#"nLepGood==2",#"nJet25_Recl<=5",
        #stop=1000000,
        stop=80000,
        process = samples["ttbar"][year]) 
    smp_tt.label_dataframe(val = 0)   #check that since this is bckg, val must be 0
    
    #df_ttw = smp_ttw.df
    df_ttbar = smp_tt.df

    #print(df_ttw)
    
    
    
    
   
    
  
    # Combinamos todos los dataframes
    dfs_to_combine = [ df_ttbar]#df_ttw,
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
    
    
    btagging(df,workingpoint="Loose")
    
    
    
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
    
    #flavouring(df_ttbar)
    #charge(df_ttbar)
    #same_mask=(df_ttbar['lep2_charge']==df_ttbar['lep1_charge']).sum()
    #diff_mask=(df_ttbar['lep2_charge']!=df_ttbar['lep1_charge']).sum()
    #print('2lss:{0}\n 2los:{1}'.format(same_mask,diff_mask))
    
    #print(df_ttbar["eventNumber"])
    #print(type(df_ttbar["eventNumber"][3]))
    #impares=(df_ttbar["eventNumber"]%2!=0).sum()
    #print("n impar={0}".format(impares))
    
    exit()
    
    
    
    
    
 ########################################				STARTING OF THE MACHINE LEARNING MODELS						#############################   
    
    
    #Variables to add to the training. We can split a diferent selection of variables for each model:
    
    vars_train_RF = ["year","htJet25j_Recl","jet2_pt","nJet25_Recl","jet3_pt","jet2_mass","lep2_pt","jet3_btagDeepFlavB",
     "lep1_pt","mlj11","jet3_mass","jet1_pt","jet4_mass","lep1_charge","lep2_mass","jet4_pt","jet3_eta","jet3_phi",
     'jet1_mass','jet4_btagDeepFlavB','Flav_muon','jet4_phi','mll','lep2_elec','jet4_eta','mlj12',"jet5_eta","Flav_elec",'mlj21','mlj22',"combipt","lep2_charge"]   #Up to here ok for the NN
     
    vars_train_NN=["year","htJet25j_Recl","jet2_pt","nJet25_Recl","jet3_pt","jet2_mass","lep2_pt","jet3_btagDeepFlavB",
     "lep1_pt","mlj11","jet3_mass","jet1_pt","jet4_mass","lep1_charge","lep2_mass","jet4_pt","jet3_eta","jet3_phi",
     'jet1_mass','jet4_btagDeepFlavB','Flav_muon','jet4_phi','mll','lep2_elec','jet4_eta','mlj12',"jet5_eta","Flav_elec",'mlj21','mlj22',"combipt","lep2_charge",   #aprox hasta aqui bn para RF
     'jet2_phi','notBjets','jet2_btagDeepFlavB', 'jet5_pt', 'jet2_eta', 'nBJetLoose25_Recl', 'MET_pt',
     "B2_pt",'B1_mass', 'B1_pt', 'B2_mass',"B1_eta","B2_eta"]
     
    vars_train_gboost=['htJet25j_Recl','jet3_pt' ,'jet2_pt' ,'year', 'jet4_pt', 'combipt', 'lep1_charge', 'lep2_mass', 'mlj12','lep2_elec',
     'lep2_pt', 'Flav_muon','B2_pt' ,'mlj11' ,'Flav_elec' ,'nBJetMedium25_Recl', 'deltarjb11' ,'jet3_eta','deltarjet', 'jet5_pt'] 
     
     
     
     
    #vars_train_RF=["notBjets","deltaetalep","deltaphilep","deltarjet","deltarlep","combipt","mll","year",
     #                    "deltaphilj11","deltaphilj12","deltaphilj21","deltaphilj22",
      #                   "deltarlj11","deltarlj12","deltarlj21","deltarlj22",
       #                  "deltarlb11","deltarlb12","deltarlb21","deltarlb22",
        #                 "deltarjb11","deltarjb12","deltarjb21","deltarjb22",
         #                "mlj11","mlj12","mlj21","mlj22",
          #               "Flav_elec","Flav_muon","Flav_mix",
           #              "B1_pt","B1_eta","B1_phi","B1_mass",
            #             "B2_pt","B2_eta","B2_phi","B2_mass",
             #            "nLepGood", 
              #           "lep1_pt", "lep1_eta","lep1_phi","lep1_mass","lep1_charge","lep1_elec",
               #          "lep2_pt", "lep2_eta","lep2_phi","lep2_mass","lep2_elec","lep2_charge",
                #         "nJet25_Recl", 
                 #        "htJet25j_Recl", 
                  #       "MET_pt", 
                   #      "nBJetLoose25_Recl",
                    #     "nBJetMedium25_Recl",
                     #    "nBJetLoose40_Recl",
                      #   "nBJetMedium40_Recl",
                       #  "jet1_pt", "jet1_eta","jet1_phi","jet1_mass","jet1_btagDeepFlavB", 
                        # "jet2_pt", "jet2_eta","jet2_phi","jet2_mass","jet2_btagDeepFlavB",
                         #"jet3_pt", "jet3_eta","jet3_phi","jet3_mass","jet3_btagDeepFlavB",
                         #"jet4_pt", "jet4_eta","jet4_phi","jet4_mass","jet4_btagDeepFlavB",
                         #"jet5_pt", "jet5_eta","jet5_phi","jet5_mass","jet5_btagDeepFlavB",
                         #"jet6_pt", "jet6_eta","jet6_phi","jet6_mass","jet6_btagDeepFlavB",
                         #"jet7_pt", "jet7_eta","jet7_phi","jet7_mass","jet7_btagDeepFlavB"]         # (A collection of all of those from one can choose)
         
    
   
   
    #		Splitting of datasets (for each of the training vars subset)
   
    X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(df[vars_train_RF], df['is_signal'], test_size=0.3, random_state=1)
    df_train_RF=pd.concat([X_train_RF,y_train_RF], axis=1)
    df_test_RF=pd.concat([X_test_RF,y_test_RF], axis=1)
    
    X_test_RF, X_validation_RF, y_test_RF, y_validation_RF = train_test_split(df_test_RF[vars_train_RF], df_test_RF['is_signal'], test_size=0.8, random_state=5)
    df_test_RF=pd.concat([X_test_RF,y_test_RF], axis=1)
    df_validation_RF=pd.concat([X_validation_RF,y_validation_RF], axis=1)
    
    
    
    X_train_NN, X_test_NN, y_train_NN, y_test_NN = train_test_split(df[vars_train_NN], df['is_signal'], test_size=0.3, random_state=1)
    df_train_NN=pd.concat([X_train_NN,y_train_NN], axis=1)
    df_test_NN=pd.concat([X_test_NN,y_test_NN], axis=1)
    
    X_test_NN, X_validation_NN, y_test_NN, y_validation_NN = train_test_split(df_test_NN[vars_train_NN], df_test_NN['is_signal'], test_size=0.8, random_state=5)
    df_test_NN=pd.concat([X_test_NN,y_test_NN], axis=1)
    df_validation_NN=pd.concat([X_validation_NN,y_validation_NN], axis=1)
    
    
    
    
    X_train_gboost, X_test_gboost, y_train_gboost, y_test_gboost = train_test_split(df[vars_train_gboost], df['is_signal'], test_size=0.3, random_state=1)
    df_train_gboost=pd.concat([X_train_gboost,y_train_gboost], axis=1)
    df_test_gboost=pd.concat([X_test_gboost,y_test_gboost], axis=1)
    
    X_test_gboost, X_validation_gboost, y_test_gboost, y_validation_gboost = train_test_split(df_test_gboost[vars_train_gboost], df_test_gboost['is_signal'], test_size=0.8, random_state=5)
    df_test_gboost=pd.concat([X_test_gboost,y_test_gboost], axis=1)
    df_validation_gboost=pd.concat([X_validation_gboost,y_validation_gboost], axis=1)
    
    
    
    
    
    
    ##### Training of the models:
    
    
    #RF=RandomForestClassifier(n_jobs=-1,min_samples_leaf=1000,max_depth=1/3*len(vars_train_RF),min_samples_split=6,n_estimators=500,max_features=10)
    #RF.fit(X_train_RF,y_train_RF)  #Trainig of the RF
    RF=RandomForestClassifier(n_jobs=-1,min_samples_leaf=5,max_depth=1/3*len(vars_train_RF),min_samples_split=2,n_estimators=100,max_features=9)
    RF.fit(X_train_RF,y_train_RF) 
    
    #Best in grid: (n_feat=,9 n_est=100, min_samp_split=2,min_samples_leaf=5) (3 decimas de overfitting y la mejora es de 0.01=>no merece la pena)
    
    
    
    gboost = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.25, n_jobs=-1,use_label_encoder=False)#ensemble.GradientBoostingClassifier(n_jobs=-1,max_depth=15, min_samples_leaf=5)				#Descomentar si quiero comparar con gradientboosting (pero solo se puede correr con pocos datos, boosting no paraleliza bien)
    gboost.fit(X_train_gboost,y_train_gboost)  #Training of the GradBoost
    
    
    
    
    model = keras.models.Sequential()
    normal_ini=keras.initializers.glorot_normal(seed=None)
    model.add(keras.layers.Dense(64,  input_shape = (len(vars_train_NN),), activation='relu'))
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
    histObj = model.fit(df_train_NN[vars_train_NN], keras.utils.to_categorical(df_train_NN["is_signal"]), epochs=25, batch_size=1000,shuffle=True,validation_data=(df_validation_NN[vars_train_NN], keras.utils.to_categorical(df_validation_NN["is_signal"])))
    #Trainig of the neural net
    
    
    
    
    
    ###############################						PLOTTING AND PRINTING RESULTS						##################################
    
    #VARIABLE IMPORTANCE
    # Variable importance of Random forest:
    
    features_list = df_train_RF.columns.values
    feature_importance = RF.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1][:20]
    sorted_idx_full=np.argsort(feature_importance)
    plt.figure(figsize=(13,10))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx][::-1], align='center')
    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx][::-1])
    plt.xlabel('Importance')
    plt.title('Feature importances')
    plt.savefig('importancia_variables_RF.png')
    #print('RF:',features_list[sorted_idx_full],feature_importance[sorted_idx_full])
    
    
    # Variable importance of Gradient Boosting:
    
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
    #print('gboost:',features_list[sorted_idx_full],feature_importance[sorted_idx_full])
    
    
    
    
   
   
    # AUCS
    #Printing AUC's of the three models
    
    ####Prediccion sobre test
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    print("Running on test sample. This may take a moment.")
    probs = model.predict(df_test_NN[vars_train_NN])#predict probability over test sample
    AUC = roc_auc_score(df_test_NN["is_signal"], probs[:,1])
    print("Test Area under Curve = {0}".format(AUC))
    
    ######Prediccion sobre train
    print("Running on train sample. This may take a moment.")
    probs2 = model.predict(df_train_NN[vars_train_NN])#predict probability over train sample
    AUC2 = roc_auc_score(df_train_NN["is_signal"], probs2[:,1])
    print("Train Area under Curve = {0}".format(AUC2))
   
    plotLearningCurves(histObj)
    
    
    print("RF AUC (test) = {0}".format(roc_auc_score(df_test_RF['is_signal'],RF.predict_proba(df_test_RF[vars_train_RF])[:,1])))
    print("RF AUC (train) = {0}".format(roc_auc_score(df_train_RF['is_signal'],RF.predict_proba(df_train_RF[vars_train_RF])[:,1])))
    print("Gboost AUC (test) = {0}".format(roc_auc_score(df_test_gboost['is_signal'],gboost.predict_proba(df_test_gboost[vars_train_gboost])[:,1])))
    print("Gboost AUC (train) = {0}".format(roc_auc_score(df_train_gboost['is_signal'],gboost.predict_proba(df_train_gboost[vars_train_gboost])[:,1])))
    
    
    
    ##Plotting the AUCs
    f, ax = plt.subplots(figsize=(7,8))   #Medida del AUC (indicador cuan bueno es el modelo) (6,6 default)
    roc_auc_plot(y_test_RF,RF.predict_proba(X_test_RF),label='Forest test',l='--')
    roc_auc_plot(y_train_RF,RF.predict_proba(X_train_RF),label='Forest train')
    
    roc_auc_plot(y_test_gboost,gboost.predict_proba(X_test_gboost),label='Gboost test',l='-')						
    roc_auc_plot(y_train_gboost,gboost.predict_proba(X_train_gboost),label='Gboost train',l='-')	
    
    roc_auc_plot(y_test_NN,model.predict(df_test_NN[vars_train_NN]),label='Net test',l='-.')
    roc_auc_plot(y_train_NN,model.predict(df_train_NN[vars_train_NN]),label='Net train')
    
    ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', label='Random Classifier')    
    ax.legend(loc="lower right")    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('Receiver Operator Characteristic curves')
    f.savefig('auc.png')    

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time/60, "minutes")
