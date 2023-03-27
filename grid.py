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


def bucle(X_train_gboost,y_train_gboost,X_test_gboost,y_test_gboost,df_test_gboost,df_train_gboost,vars_train_gboost):
	best_auc=0
	best_lr=0
	best_depth=0
	best_subsample=0
	lr=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
	max_depth=[2,3,5,10]
	subsample=[0.5,0.6,0.7,0.8,1]
	for i in max_depth:
		for j in lr:
			for k in subsample:
				gboost=xgb.XGBClassifier(n_estimators=100,eval_metric='logloss',subsample=k,max_depth=i,learning_rate=j,n_jobs=-1,use_label_encoder=False)
				gboost.fit(X_train_gboost,y_train_gboost)
				print("Max_depth={0}\n Learning_rate={1}\n Subsample={2}".format(i,j,k))
				print("Gboost AUC (test) = {0}".format(roc_auc_score(df_test_gboost['is_signal'],gboost.predict_proba(df_test_gboost[vars_train_gboost])[:,1])))
				print("Gboost AUC (train) = {0}".format(roc_auc_score(df_train_gboost['is_signal'],gboost.predict_proba(df_train_gboost[vars_train_gboost])[:,1])))
				if roc_auc_score(df_test_gboost['is_signal'],gboost.predict_proba(df_test_gboost[vars_train_gboost])[:,1])>best_auc:
					best_auc=roc_auc_score(df_test_gboost['is_signal'],gboost.predict_proba(df_test_gboost[vars_train_gboost])[:,1])
					best_lr=j
					best_depth=i
					best_subsample=k

	print("Best AUC ({0}) achieved with learning rate={1}, max_depth={2} and subsample={3}".format(best_auc,best_lr,best_depth,best_subsample))

def bucle_RF(X_train_RF,y_train_RF,X_test_RF,y_test_RF,df_test_RF,df_train_RF,vars_train_RF):
	best_auc=0
	best_nest=0
	best_features=0
	best_samp_split=0
	best_samp_leaf=0
	nest=[100,200,300,400,500]
	features=[5,6,8,9,10]
	samp_split=[2,4,6,8,10]
	samp_leaf=[1,2,3,4,5]
	for i in features:
		for j in nest:
			for k in samp_split:
				for h in samp_leaf:
					RF=RandomForestClassifier(n_jobs=-1,min_samples_leaf=h,max_depth=1/3*len(vars_train_RF),min_samples_split=k,n_estimators=j,max_features=i)
					RF.fit(X_train_RF,y_train_RF)
					print("N_features={0}\n N_estimators={1}\n Min_samples_split={2}\n Min_samples_leaf={3}".format(i,j,k,h))
					print("RF AUC (test) = {0}".format(roc_auc_score(df_test_RF['is_signal'],RF.predict_proba(df_test_RF[vars_train_RF])[:,1])))
					print("RF AUC (train) = {0}".format(roc_auc_score(df_train_RF['is_signal'],RF.predict_proba(df_train_RF[vars_train_RF])[:,1])))
					if roc_auc_score(df_test_RF['is_signal'],RF.predict_proba(df_test_RF[vars_train_RF])[:,1])>best_auc:
						best_auc=roc_auc_score(df_test_RF['is_signal'],RF.predict_proba(df_test_RF[vars_train_RF])[:,1])
						best_nest=j
						best_features=i
						best_samp_split=k
						best_samp_leaf=h

	print("Best AUC ({0}) achieved with N_features={1}\n N_estimators={2}\n Min_samples_split={3}\n Min_samples_leaf={4}".format(best_auc,best_features,best_nest,best_samp_split,best_samp_leaf))
