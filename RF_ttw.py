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

#pd.set_option('display.max_rows',10)
#pd.set_option('display.max_columns',None)


def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))


if __name__ == "__main__":
    mainpath = "/beegfs/data/TOPnanoAODv6/ttW_MC_Ntuples_skim/"

    # Dictionary with sample names
    samples = {
      "ttbar" : { 2016 : "TTLep_pow_part1", #Se mpueden meter listas
                  2017 : "TTLep_pow",
                  2018 : "TTLep_pow"}, 
      "ttW"   : { 2016 : "TTWToLNu_PSW", 
                  2017 : "TTWToLNu_PSW",
                  2018 : "TTWToLNu_fxfx"}
    }
    # -- Variables to read from trees (can be defined in a friend tree) 
    friends = ["1_recl_enero"]   
    branches = ["year", 
                "nLepGood", 
                "LepGood_pt[0]", "LepGood_eta[0]","LepGood_phi[0]","LepGood_mass[0]",
                "LepGood_pt[1]", "LepGood_eta[1]","LepGood_phi[1]","LepGood_mass[1]",
                "JetSel_Recl_pt[0]", "JetSel_Recl_eta[0]","JetSel_Recl_phi[0]","JetSel_Recl_mass[0]",
                "JetSel_Recl_pt[1]", "JetSel_Recl_eta[1]","JetSel_Recl_phi[1]","JetSel_Recl_mass[1]",
                "JetSel_Recl_pt[2]", "JetSel_Recl_eta[2]","JetSel_Recl_phi[2]","JetSel_Recl_mass[2]",
                #"JetSel_Recl_pt[2]","JetSel_Recl_pt[3]","JetSel_Recl_pt[4]",
                "nJet25_Recl", "htJet25j_Recl", 
                "MET_pt", 
                "nBJetLoose25_Recl","nBJetMedium25_Recl", "nBJetLoose40_Recl","nBJetMedium40_Recl",
                "LepGood_jetBTagDeepFlav[0]","LepGood_jetBTagDeepFlav[1]",
                "JetSel_Recl_btagDeepFlavB[0]","JetSel_Recl_btagDeepFlavB[1]","JetSel_Recl_btagDeepFlavB[2]"]
    
    
    
    
    # Create the signal dataframe
    smp_ttw = smp2df(branches = branches,  #sample2df comvierte sample a data frame
                     friends = friends, 
                     name = "df_ttw")
    
    
            
    for year in [2016, 2017, 2018]:
      smp_ttw.load_data(                   #loaddata en samples.py
        path = mainpath + "%s"%year, 
        selection = "nJet25_Recl==3",  #&(JetSel_Recl_btagDeepFlavB[0]>=0.5)",    #Selection to add in case we deal with certain requirement (for instance if we add a variable on jets, we must ask for having at least one jet)
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
        selection = "nJet25_Recl==3",
        stop=1000000,
        process = samples["ttbar"][year]) 
    smp_tt.label_dataframe(val = 0)   #check that since this is bckg, val must be 0
    
    df_ttw = smp_ttw.df
    df_ttbar = smp_tt.df
   
   
   #Dummy values:
    #df_ttw["JetSel_Recl_btagDeepFlavB[1]"]=np.where(df_ttw["JetSel_Recl_btagDeepFlavB[1]"]<=0.03,-99,df_ttw["JetSel_Recl_btagDeepFlavB[1]"])
    #df_ttbar["JetSel_Recl_btagDeepFlavB[1]"]=np.where(df_ttbar["JetSel_Recl_btagDeepFlavB[1]"]<=0.03,-99,df_ttbar["JetSel_Recl_btagDeepFlavB[1]"])
	

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
    
    
    btagging(df)
    print(df)		
	    
	    
	    
    
    
	
    
    
    ############################### Definicion variables 'complejas'
    
    # Create 4 vectors
    l1 = create4vec(df["LepGood_pt[0]"], df["LepGood_eta[0]"], df["LepGood_phi[0]"], df["LepGood_mass[0]"])
    l2 = create4vec(df["LepGood_pt[1]"], df["LepGood_eta[1]"], df["LepGood_phi[1]"], df["LepGood_mass[1]"])
    
    j1 = create4vec(df["JetSel_Recl_pt[0]"], df["JetSel_Recl_eta[0]"], df["JetSel_Recl_phi[0]"], df["JetSel_Recl_mass[0]"])
    j2 = create4vec(df["JetSel_Recl_pt[1]"], df["JetSel_Recl_eta[1]"], df["JetSel_Recl_phi[1]"], df["JetSel_Recl_mass[1]"])
    
    b1=create4vec(df["B1_pt"],df["B1_eta"],df["B1_phi"],df["B1_mass"])
    
    
    df["mll"] = mll(l1, l2)
    df["combipt"]=combipt(l1,l2)
    df["deltarlep"]=deltar(l1,l2)
    df["deltarjet"]=deltar(j1,j2)
    df["deltarlj"]=deltar(l1,j1)
    df["deltarblep"]=deltar(b1,l1)
    df["deltarbj"]=deltar(b1,j1)
    
    #Variables to add to the training
    vars_train = ["LepGood_pt[0]","LepGood_pt[1]","JetSel_Recl_pt[0]","MET_pt","mll","deltarjet","htJet25j_Recl","deltarlep",
    "JetSel_Recl_btagDeepFlavB[0]","JetSel_Recl_btagDeepFlavB[1]","deltarlj","LepGood_jetBTagDeepFlav[0]","deltarblep","deltarbj"]
     #vars_train=["mll","deltarlep","deltarjet","combipt",
     #           "year", 
      #          "nLepGood", 
       #         "LepGood_pt[0]", "LepGood_eta[0]","LepGood_phi[0]","LepGood_mass[0]",
        #        "LepGood_pt[1]", "LepGood_eta[1]","LepGood_phi[1]","LepGood_mass[1]",
         #       "JetSel_Recl_pt[0]", "JetSel_Recl_eta[0]","JetSel_Recl_phi[0]","JetSel_Recl_mass[0]",
          #      "JetSel_Recl_pt[1]", "JetSel_Recl_eta[1]","JetSel_Recl_phi[1]","JetSel_Recl_mass[1]",
           #     "nJet25_Recl", "htJet25j_Recl", 
            #    "MET_pt", 
             #   "nBJetLoose25_Recl","nBJetMedium25_Recl", "nBJetLoose40_Recl","nBJetMedium40_Recl"]#,
                #"LepGood_jetBTagDeepFlav[0]","LepGood_jetBTagDeepFlav[1]"]
    
    
    # From Andrea
    X_train, X_test, y_train, y_test = train_test_split(df[vars_train], df['is_signal'], test_size=0.3, random_state=1)
    df_train=pd.concat([X_train,y_train], axis=1)
    df_test=pd.concat([X_test,y_test], axis=1)
    
    X_test, X_validation, y_test, y_validation = train_test_split(df_test[vars_train], df_test['is_signal'], test_size=0.8, random_state=5)
    df_test=pd.concat([X_test,y_test], axis=1)
    df_validation=pd.concat([X_validation,y_validation], axis=1)
    
    
    RF=RandomForestClassifier(n_jobs=-1)#min_samples_leaf=5,max_depth=15,min_samples_split=2)
    RF.fit(X_train,y_train)  #Trainig of the RF
    
    
    
    
    
    
    
    model = keras.models.Sequential()
    normal_ini=keras.initializers.glorot_normal(seed=None)
    model.add(keras.layers.Dense(64,  input_shape = (len(vars_train),), activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    #model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.summary()
    
    sgd = keras.optimizers.SGD(lr=0.01)
    #adamax=keras.optimizers.Adamax(learning_rate=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    adam=keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics = ["accuracy"])
    histObj = model.fit(df_train[vars_train], keras.utils.to_categorical(df_train["is_signal"]), epochs=20, batch_size=1000,validation_data=(df_validation[vars_train], keras.utils.to_categorical(df_validation["is_signal"])))
    
    
    
    
    
    train=df_train    							#Code for printing the relative importance of variables
    features_list = train.columns.values
    feature_importance = RF.feature_importances_
    sorted_idx = np.argsort(feature_importance)[:20]
    plt.figure(figsize=(10,7))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature importances')
    plt.savefig('importancia_variables_RF.png')
    
   
    
    
    
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
    
    plotLearningCurves(histObj)
    
    
    #gboost = ensemble.GradientBoostingClassifier(max_depth=15, min_samples_leaf=5)				#Descomentar si quiero comparar con gradientboosting (pero solo se puede correr con pocos datos, boosting no paraleliza bien)
    #gboost.fit(X_train,y_train)
    #feature_importance = gboost.feature_importances_
    #sorted_idx = np.argsort(feature_importance)[:20]
    #plt.figure(figsize=(10,7))
    #plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    #plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
    #plt.xlabel('Importance')
    #plt.title('Feature importances')
    #plt.savefig('importancia_variables_Gboost.png')
    
    
    
    
    
    f, ax = plt.subplots(figsize=(6,6))   #Medida del AUC (indicador cuan bueno es el modelo)
    roc_auc_plot(y_test,RF.predict_proba(X_test),label='FOREST ',l='--')
    #roc_auc_plot(y_test,gboost.predict_proba(X_test),label='GBOOST',l='-')						#Descomentar si gboost
    roc_auc_plot(y_test,model.predict(df_test[vars_train]),label='NN',l='-.')
    ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', label='Random Classifier')    
    ax.legend(loc="lower right")    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('Receiver Operator Characteristic curves')
    f.savefig('auc.png')    
