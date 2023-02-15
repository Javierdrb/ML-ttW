# To save useful functions to use both for the RF and DNN
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
from sklearn import tree, ensemble
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tensorflow import keras

@np.vectorize
def create4vec(pt,eta,phi,m):
	vec=r.TLorentzVector()
	vec.SetPtEtaPhiM(pt, eta, phi, m)
	return vec 

@np.vectorize
def mll(p1, p2):
	return (p1+p2).M() 

@np.vectorize
def combipt(p1,p2):
	return p1.Pt()+p2.Pt()
	
@np.vectorize
def deltaphi(p1,p2):
	return abs(p1.Phi()-p2.Phi())

@np.vectorize
def deltaeta(p1,p2):
	return abs(p1.Eta()-p2.Eta())

@np.vectorize
def deltar(p1,p2):
	return np.sqrt(deltaphi(p1,p2)**2+deltaeta(p1,p2)**2)

def btagging(df,workingpoint="Medium"):   #Function  to identify which is the first b-jet and create associated columns for pt,eta,phi,mass (copying those of the jet that corresponds)
	#STILL WORKING IN IT:
						#Different values in working point depending on the year
						#Written just for a fixed value of jets
    njets = df["nJet25_Recl"].values.astype(int)
    #btag_scores = df.loc[:, ["JetSel_Recl_btagDeepFlavB[{}]".format(j) for j in range(njets.max())]].values
    
    btag_scores = df.loc[:, ["jet{}_btagDeepFlavB".format(j) for j in range(1,njets.max()+1)]].values
    year=df.loc[:,["year"]].values
    

  
    if workingpoint == "Medium":
        is_btagged = ((btag_scores >= 0.3093) & (year == 2016)) | ((btag_scores >= 0.3033) & (year == 2017)) | ((btag_scores >= 0.2770) & (year == 2018))   
    elif workingpoint == "Loose":
        is_btagged = ((btag_scores >= 0.0614) & (year == 2016)) | ((btag_scores >= 0.0521) & (year == 2017)) | ((btag_scores >= 0.0494) & (year == 2018))
    else:
        is_btagged = ((btag_scores >= 0.7221) & (year == 2016)) | ((btag_scores >= 0.7489) & (year == 2017)) | ((btag_scores >= 0.7264) & (year == 2018))
    
    
    btagged_jets = np.apply_along_axis(np.argmax, axis=1, arr=is_btagged)
    btagged_jets[~is_btagged.any(axis=1)] = -1
   
    # Get the indexes of 1st btagged jet in each row
    mask=(btagged_jets>=0)
    idx=btagged_jets[mask]
    
    # Select the pt, eta, phi, and mass columns for the first b-tagged jet for each row
    # Create an array of dummy values with the same shape as `B1_pt`
    B1_pt = np.repeat(-99.0,df.shape[0])
    B1_eta = np.repeat(-99.0,df.shape[0])
    B1_phi = np.repeat(-99.0,df.shape[0])
    B1_mass = np.repeat(-99.0,df.shape[0])
    
    # Replace the appropriate values in `B1_pt`    ####   NOTE: it is prepared for 5 jets, if this is different, it must be fixed
    B1_pt[mask] = np.where(idx == 0, df.loc[mask, "jet1_pt"],
     np.where(idx == 1, df.loc[mask, "jet2_pt"],np.where(idx == 2, df.loc[mask, "jet3_pt"], np.where(idx == 3, df.loc[mask, "jet4_pt"],
     df.loc[mask, "jet5_pt"]))))
    B1_eta[mask] = np.where(idx == 0, df.loc[mask, "jet1_eta"],
     np.where(idx == 1, df.loc[mask, "jet2_eta"],np.where(idx == 2, df.loc[mask, "jet3_eta"], np.where(idx == 3, df.loc[mask, "jet4_eta"],
     df.loc[mask, "jet5_eta"]))))
    B1_phi[mask] = np.where(idx == 0, df.loc[mask, "jet1_phi"],
     np.where(idx == 1, df.loc[mask, "jet2_phi"],np.where(idx == 2, df.loc[mask, "jet3_phi"], np.where(idx == 3, df.loc[mask, "jet4_phi"],
     df.loc[mask, "jet5_phi"]))))
    B1_mass[mask] = np.where(idx == 0, df.loc[mask, "jet1_mass"],
     np.where(idx == 1, df.loc[mask, "jet2_mass"],np.where(idx == 2, df.loc[mask, "jet3_mass"], np.where(idx == 3, df.loc[mask, "jet4_mass"],
     df.loc[mask, "jet5_mass"]))))
    
     # Replace the `B1_pt` column in `df`
    df["B1_pt"] = B1_pt
    df["B1_eta"] = B1_eta
    df["B1_phi"] = B1_phi
    df["B1_mass"] = B1_mass

    return df
	
#def btagging(df):      #Less efficient b-tagging (with high amount of data even does not work)
#	nrows=df.shape[0]
#	for i in range(nrows): #nrows
#		njets=df["nJet25_Recl"][i].astype(int)
#		btag=[]
#		nobtag=[]
		#print("nJets:",njets)
#		for j in range(njets):
			#print(j,df["JetSel_Recl_btagDeepFlavB[{}]".format(j)][i])
			
#			if df["JetSel_Recl_btagDeepFlavB[{}]".format(j)][i]>=0.3093:
#				btag.append(j)
#			else:
#				nobtag.append(j)
		#print("Btagged in event {}".format(i),btag,"Non-btagged in event {}".format(i),nobtag)
#		if btag:
#			df.loc[i,"B1_pt"]=df["JetSel_Recl_pt[{}]".format(btag[0])][i]
#			df.loc[i,"B1_eta"]=df["JetSel_Recl_eta[{}]".format(btag[0])][i]
#			df.loc[i,"B1_phi"]=df["JetSel_Recl_phi[{}]".format(btag[0])][i]
#			df.loc[i,"B1_mass"]=df["JetSel_Recl_mass[{}]".format(btag[0])][i]
				
#		else:
#			df.loc[i,"B1_pt"]=-99
#			df.loc[i,"B1_eta"]=-99
#			df.loc[i,"B1_phi"]=-99
#			df.loc[i,"B1_mass"]=-99
	

#def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
 #   from sklearn.metrics import roc_curve, roc_auc_score
 #   fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
 #   ax.plot(fpr, tpr, linestyle=l, linewidth=lw,label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))

def plotLearningCurves(*histObjs):
    """This function processes all histories given in the tuple.
    Left losses, right accuracies
    """
    # too many plots
    if len(histObjs)>10: 
        print('Too many objects!')
        return
    # missing names
    for histObj in histObjs:
        if not hasattr(histObj, 'name'): histObj.name='?'
    names=[]
    # loss plot
    plt.figure(figsize=(12,6))
    plt.rcParams.update({'font.size': 15}) #Larger font size
    plt.subplot(1,2,1)
    # loop through arguments
    for histObj in histObjs:
        plt.plot(histObj.history['loss'])
        names.append('train '+histObj.name)
        plt.plot(histObj.history['val_loss'])
        names.append('validation '+histObj.name)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper right')
    

    #accuracy plot
    plt.subplot(1,2,2)
    for histObj in histObjs:
        plt.plot(histObj.history['accuracy'])
        plt.plot(histObj.history['val_accuracy'])
    plt.title('model accuracy')
    #plt.ylim(0.5,1)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(names, loc='upper left')
    
    plt.savefig("evolNN.png")
    
    # min, max for loss and acc
    for histObj in histObjs:
        h=histObj.history
        maxIdxTrain = np.argmax(h['accuracy'])
        maxIdxTest  = np.argmax(h['val_accuracy'])
        minIdxTrain = np.argmin(h['loss'])
        minIdxTest  = np.argmin(h['val_loss'])
        
        strg='\tTrain: Min loss {:6.3f} at {:3d} --- Max acc {:6.3f} at {:3d} | '+histObj.name
        print(strg.format(h['loss'][minIdxTrain],minIdxTrain,h['accuracy'][maxIdxTrain],maxIdxTrain))
        strg='\tValidation : Min loss {:6.3f} at {:3d} --- Max acc {:6.3f} at {:3d} | '+histObj.name
        print(strg.format(h['val_loss'][minIdxTest],minIdxTest,h['val_accuracy'][maxIdxTest],maxIdxTest))
        print(len(strg)*'-')



