import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')

def get_imp_features(X_train,Y_train, no_of_features = 10) :
	all_core = []
	feature_name = X_train.columns.tolist()
	for col in X_train.columns.tolist() : 
		coef = np.corrcoef(X_train[col],Y_train)[0,1]
		all_core.append(coef)
	all_core = [0 if np.isnan(i) else i for i in all_core]
	cor_feature = X_train.iloc[:,np.argsort(np.abs(all_core))[-1*no_of_features:]].columns.tolist()
	cor_support = [True if i in cor_feature else False for i in feature_name]
	#cor_feature, cor_support  = get_cor_feature(X_train,Y_train)

	X_Norm = MinMaxScaler().fit_transform(X_train)
	chi_selector = SelectKBest(chi2, k = no_of_features)
	chi_selector.fit(X_Norm, Y_train)
	chi_support = chi_selector.get_support()
	chi_feature = X_train.loc[:,chi_support].columns.tolist()
	#print(str(len(chi_feature)), 'selected features')
	
	#####################
	# RFE Estimator
	#####################
	rfe_selector = RFE(estimator = LogisticRegression(), n_features_to_select = no_of_features, step = 4, verbose = 5)
	rfe_selector.fit(X_train,Y_train)
	rfe_feature_support = rfe_selector.get_support()
	rfe_selected_feature = X_train.loc[:,rfe_feature_support].columns.tolist()
	#print(str(len(rfe_selected_feature)), 'Selected features')
	
	#####################
	# Logistics Regression
	#####################
	lr_selector = SelectFromModel(LogisticRegression(penalty = "l1", solver='liblinear'),'1.25*median')
	lr_selector.fit(X_train,Y_train)
	lr_feature_support = lr_selector.get_support()
	lr_selected_feature = X_train.loc[:,lr_feature_support].columns.tolist()
	#print(str(len(lr_selected_feature)), 'Selected features')
	
	#####################
	# Random Forest Estimator
	#####################
	rf_selector = SelectFromModel(RandomForestClassifier(n_estimators = 100),'1.25*median')
	rf_selector.fit(X_train,Y_train)
	rf_feature_support = rf_selector.get_support()
	rf_selected_feature = X_train.loc[:,rf_feature_support].columns.tolist()
	#print(str(len(rf_selected_feature)), 'Selected features')
	
	
	#####################
	# LGB Estimator
	#####################
	#lgb_model = LGBMClassifier(n_estimator = 500, learning_rate = 0.05, num_leaves = 32, colsample_btree = 0.2,
	#                         reg_alpha = 3, reg_lambda = 1, min_split_gain=0.01, min_child_weight=40)
	lgb_model = LGBMClassifier(n_estimator = 500, learning_rate = 0.05, num_leaves = 32)
	lgb_selector = SelectFromModel(lgb_model,'1.25*median')
	lgb_selector.fit(X_train,Y_train)
	lg_feature_support = lgb_selector.get_support()
	lg_selected_feature = X_train.loc[:,lg_feature_support].columns.tolist()
	#print(str(len(lg_selected_feature)), 'Selected features')
	
	pd.set_option('display.max_rows', None)
	# put all selection together
	feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support,
										'RFE':rfe_feature_support, 'Logistics':lr_feature_support,
										'Random Forest':rf_feature_support, 'LightGBM':lg_feature_support})
	feature_selection_df.shape
	feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
	feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
	feature_selection_df.index = range(1, len(feature_selection_df)+1)
	return(feature_selection_df)