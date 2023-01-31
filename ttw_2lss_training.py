''' Implementation of a neural network to discriminate ttW '''
# -- Import libraries 
import pandas as pd
import numpy as np
import ROOT as r
import df_utils as dfu
from samples import smp2df
import matplotlib.pyplot as plt

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
from tensorflow import keras


if __name__ == "__main__":
    mainpath = "/beegfs/data/TOPnanoAODv6/ttW_MC_Ntuples_skim/"

    # Dictionary with sample names
    samples = {
      "ttbar" : { 2016 : "TTLep_pow_part1", 
                  2017 : "TTLep_pow",
                  2018 : "TTLep_pow"}, 
      "ttW"   : { 2016 : "TTWToLNu_PSW", 
                  2017 : "TTWToLNu_PSW",
                  2018 : "TTWToLNu_fxfx"}
    }
    # -- Variables to read from trees (can be defined in a friend tree) 
    friends = ["1_recl_enero"]
    branches = ["year", "nLepGood", "nJet25_Recl", "LepGood_pt[0]"]
    
    # Create the signal dataframe
    smp_ttw = smp2df(branches = branches, 
                     friends = friends, 
                     name = "df_ttw")
    for year in [2016, 2017, 2018]:
      smp_ttw.load_data(
        path = mainpath + "%s"%year, 
        process = samples["ttW"][year]) 
    smp_ttw.label_dataframe(val = 1)   

    # Create the bkg dataframe
    smp_tt = smp2df(branches = branches, 
                    friends = friends, 
                    name = "df_tt")
    for year in [2016, 2017, 2018]:
      smp_tt.load_data(
        path = mainpath + "%s"%year, 
        process = samples["ttbar"][year]) 
    smp_tt.label_dataframe(val = 0)
    
    df_ttw = smp_ttw.df
    df_ttbar = smp_tt.df   

#    df_ttw = np.hstack(df_ttw)
    print(df_ttw)
    # Combinamos todos los dataframes
    vars_train = ["year", "nLepGood", "nJet25_Recl"]
    dfs_to_combine = [df_ttw, df_ttbar]
    df = dfu.combine_dataframes(dfs_to_combine, axis = 0) # Para concatenar filas
    print(df)

    # From Andrea
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
