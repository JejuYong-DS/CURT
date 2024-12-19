'''
Title : ,,,

author : Juyong Ko
Department of Big Data Science, College of Public Policy, Korea University

2023_2R CURT Code
'''
#%% Set Seed
import random
random.seed(0)
random_state = 0

#%% module
# !pip install factor_analyzer
# !pip install imblearn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import time
#%% 해상도 세팅
#plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
#sns
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")

#%% Function
#값의 형태 확인
def unique_preprocess(df): 
    print('---remove---')
    for i in df.columns:
        #Unique()가 1인 변수 제거
        if len(df[i].unique()) == 1:
            print(i)
            df = df.drop(i, axis = 1)
    return df

#VIF 계산
def vif_calculate(df):
    vif = pd.DataFrame()
    vif['col_name'] = df.columns
    vif['VIF'] = [format(variance_inflation_factor(df.values, i),'f') for i in range(len(df.columns))]
    return vif

#요인분석
def factor_analysis(n, df_FA, method=None):
    # n : 요인 수, df_FA : dataframe, method : rotation method
    fa = FactorAnalyzer(n_factors=n, rotation=method)
    fa.fit(df_FA)
    df_FA_loadings = pd.DataFrame(fa.loadings_, index=df_FA.columns)
    
    return df_FA_loadings, fa

#요인분석 히트맵
def FA_heatmap(df_FA_loadings, title=None):
    plt.figure(figsize=(15,20))
    sns.heatmap(df_FA_loadings,
                cmap="RdBu",
                annot=True,
                fmt='.2f',
                vmax=1, vmin=-1)
    plt.title(title, fontsize = 20)
    plt.xlabel('Factor', fontsize = 15)
    link_title = os.path.join(link, title)
    plt.savefig(f'{link_title}.png', bbox_inches='tight')
    plt.show()

# Scree Plot
def scree_plot(x,y,title=None):
    plt.plot(x, y, 'o-',  color='black')
    plt.title('Scree Plot')
    plt.xlabel('Number of principal components')
    plt.ylabel('Variance Explained')
    plt.title(f'<{title}>')
    link_title = os.path.join(link, title)
    plt.savefig(f'{link_title}.png')
    plt.show()    

#%% 1. raw data
#Call dataset
link = r'C:\Users\rhwnd\OneDrive\바탕 화면\CDS\[232R] 2023 2R Curt'
os.chdir(link)
df = pd.read_csv(link+'\data.csv')
y = df['Bankrupt?']

#Unique Preprocess : Remove a feature with zero variation.
df = unique_preprocess(df)

# remove duplication
df = df.loc[:, ~df.T.duplicated()]

#missing values check
print('missing values check :', df.isnull().sum().sum())

#Descriptive Statistics
discribe_df = df.describe()
print(discribe_df)
# discribe_df.to_html(link+'\discribe_df.html')

#Scailing
scaler = StandardScaler()
df_raw = pd.DataFrame(scaler.fit_transform(df.iloc[:,1:]), columns = df.columns[1:])

#save file
# df_raw.to_json(link+'\df_raw.json')

#Descriptive Statistics
discribe_df_raw = df_raw.describe()
print(discribe_df_raw)
# discribe_df_raw.to_html(link+'\discribe_df_raw.html')

#%% 2. VIF
df_VIF = df_raw.copy()

vif = vif_calculate(df_VIF)
print(vif)
vif.to_html(link+r'\VIF.html')

#if VIF > 10, list-up
VIF_over10_list = [vif['col_name'][i] for i in range(len(vif)) if float(vif['VIF'][i]) > 10]

#Remove
df_VIF = df_VIF.drop(VIF_over10_list, axis = 1)

#save file
# df_VIF.to_json(link+'\df_VIF.json')

discribe_df_VIF = df_VIF.describe()
# discribe_df_VIF.to_html(link+'\discribe_df_VIF.html')


#%% 3. FA
df_FA = df_VIF.copy()

#바틀렛 검정
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df_FA)
print(chi_square_value, p_value) #p<.001

#KMO 검정
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all, kmo_model = calculate_kmo(df_FA)
print(round(kmo_model, 2)) #0.64

#요인 수 선택
fa = FactorAnalyzer(n_factors=len(df_FA.columns), rotation=None)
fa.fit(df_FA)

#Kaiser's Rule
ev,v = fa.get_eigenvalues() # print(ev)
ev_1 = [i for i in ev if i >= 1] # ev >= 1

#rotation을 None
df_FA_loadings, fa = factor_analysis(len(ev_1), df_FA, method=None)
FA_heatmap(df_FA_loadings, title='Without Applying Varimax')

#orthogonal rotation인 varimax 선택
df_FA_loadings_varimax, fa_varimax = factor_analysis(len(ev_1), df_FA, method='varimax')
FA_heatmap(df_FA_loadings_varimax, title='With Applying Varimax')

#분석 결과
fa_varimax.get_factor_variance() #누적 분산 0.451272
df_FA_variance = pd.DataFrame(fa_varimax.get_factor_variance(), index=['SS Loadings', 'Proportion Var', 'Cumulative Var'])

#save file
df_FA_variance.to_html(link+r'\df_FA_variance.html')

#K = 0.3 이상의 적재량(절댓값)이 한번도 안나온 변수 제거
K = 0.3
loadings_under_list = []
for i in range(len(df_FA_loadings_varimax)):
    loadings_under = [abs(k) >= K for k in df_FA_loadings_varimax.iloc[i]]
    if sum(loadings_under) == 0:
        loadings_under_list.append(df_FA_loadings_varimax.index[i])

df_FA = df_FA.drop(loadings_under_list, axis = 1)
#save file
df_FA.to_json(link+r'\df_FA.json')

#%% 4. PCA
df_PCA = df_raw.copy()

#최적의 n_components 찾기
from sklearn.decomposition import PCA
pca = PCA(n_components=len(df_PCA.columns))
pca.fit(df_PCA)

#Scree Plot의 X, y
PC_values = np.arange(pca.n_components) + 1
ve = pca.explained_variance_ratio_

#Scree Plot
scree_plot(PC_values, ve, title='Scree Plot')

# Kaiser's Rule : eigen value >= 1
import numpy.linalg as lin
feature = df_raw.T
cov_matrix = np.cov(feature)
print(lin.eig(cov_matrix)[0])
# 29

#n = 29
pca = PCA(n_components=29)
pca.fit(df_PCA)

#Scree Plot의 X, y
PC_values = np.arange(pca.n_components) + 1
ve = pca.explained_variance_ratio_

#Scree plot
scree_plot(PC_values, ve, title='Scree 29')
# print(ve)
print('Cumulative Var :', round(sum(ve), 2))

#pca로 이루어진 DataFrame 생성
pca_columns_list = ['PC'+str(i) for i in PC_values]
df_PCA = pd.DataFrame(data=pca.fit_transform(df_PCA), columns = pca_columns_list)

#save file
df_PCA.to_json(link+r'\df_PCA.json')

#Descriptive Statistics
discribe_df_PCA = df_PCA.describe()
discribe_df_PCA.to_html(link+r'\discribe_df_PCA.html')
#%%
PC_coef = pd.DataFrame(pca.components_.T, index = df_raw.columns)
plt.figure(figsize=(15,20))
sns.heatmap(PC_coef,
            cmap="RdBu",
            annot=True,
            fmt='.2f',
            vmax=1, vmin=-1)
#%% Analysis
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold


#UnderSampling
def UnderSampling(X, y):
    rus = RandomUnderSampler(random_state=random_state)
    X, y = rus.fit_resample(X, y)
    return X, y

def Analysis(X,y, model, option=None, random_state = random_state):
    """
    option : 'evaluation', 'cv', 'roc'
    """
    
    #Hold_Out
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)
    
    #Modeling
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    if option == 'evaluation':
        #Odds Ratio
        [[TP,FN], [FP,TN]] = confusion_matrix(y_test,y_pred)
        
        Odds_Ratio = (TP*TN)/(FN*FP)
        print('Confusion Matrix :\n', confusion_matrix(y_test,y_pred))
        print('Odds Ratio :', round(Odds_Ratio, 6))

        #모델 평가 지표 (evaluation metrics for classification model)
        
        AUC = round(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]),6)
        Accuracy = round(accuracy_score(y_test,y_pred),6)
        Precision = round(precision_score(y_test,y_pred),6)
        Recall = round(recall_score(y_test,y_pred),6)
        F1 = round(f1_score(y_test,y_pred),6)
        
        return [AUC, Accuracy, Precision, Recall, F1]
    
    if option == 'cv':
        cv = RepeatedKFold(n_splits=5, n_repeats=100, random_state=random_state)
        cv_5 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
        return cv_5
        # cv_5 = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        # cv_10 = cross_val_score(model, X, y, cv=10, scoring='f1_macro')
        # return (np.mean(cv_5), abs(np.min(cv_5)-np.mean(cv_5)), abs(np.max(cv_5)-np.mean(cv_5))), (np.mean(cv_10), abs(np.min(cv_10)-np.mean(cv_10)), abs(np.max(cv_10)-np.mean(cv_10)))
    if option == 'roc':
        pred_positive_label = model.predict_proba(x_test)[:,1]
        fprs, tprs, thresholds = roc_curve(y_test, pred_positive_label)
        auc_score = roc_auc_score(y_test, y_pred)
        return fprs, tprs, thresholds, auc_score
    
    return y_test, y_pred

#%% Model fitting
dfs = {'Raw Data':df_raw, 'Applying VIF':df_VIF, 'Applying FA':df_FA, 'Applying PCA':df_PCA}
models = {}

from sklearn.linear_model import LogisticRegression
models['LR'] = LogisticRegression(random_state=random_state)

from sklearn import svm
models['SVM'] = svm.SVC(kernel='linear', C=1, random_state=random_state, probability=True)

from sklearn.ensemble import RandomForestClassifier
models['RF'] = RandomForestClassifier(max_depth=5, random_state=random_state)
# models['RF2'] = RandomForestClassifier(random_state=random_state)

from sklearn.tree import DecisionTreeClassifier
models['DT'] = DecisionTreeClassifier(random_state=random_state)

from sklearn.linear_model import SGDClassifier
models['SGD'] = SGDClassifier(loss="log_loss", random_state=random_state)

from sklearn.ensemble import GradientBoostingClassifier
models['GB'] = GradientBoostingClassifier(random_state=random_state)

from sklearn.gaussian_process import GaussianProcessClassifier
models['GP'] = GaussianProcessClassifier(random_state=random_state)

from sklearn.neural_network import MLPClassifier
models['MLP'] = MLPClassifier(solver='lbfgs', random_state=random_state)

from xgboost import XGBClassifier
models['XGB'] = XGBClassifier(random_state=random_state)

#%% evaluation
score_col_list = []
evaluation_list = []
score_df = pd.DataFrame()
for df_name, df_temp in dfs.items():
    print(df_name)
    for model_name, model in models.items():
        print(model_name)
        X, y = UnderSampling(df_temp, df['Bankrupt?'])
        score_col_list.append((df_name,model_name))
        evaluation_list.append(Analysis(X, y, model, option='evaluation'))
        
score_df = pd.DataFrame(evaluation_list, index = score_col_list, columns = ['AUC', 'Accuracy', 'Precision', 'Recall', 'f1 score'])
score_df = score_df.reindex(pd.MultiIndex.from_tuples(score_col_list))
score_df.to_html(link+r'\score_df.html')
#%%
df_f1_score = score_df['f1 score'].reset_index().pivot(index = 'level_0', columns = 'level_1', values = 'f1 score')
df_f1_score.index.name = ''
df_f1_score.columns = models.keys()
df_f1_score.to_html(link+r'\f1_scores.html')

#%% cv
from sklearn.model_selection import cross_val_score

# cv_5_list = []
# cv_10_list = []
cv_5_dict = {}
for df_name, df_temp in dfs.items():
    if df_name == 'Raw Data':
        continue
    print('-'*20, df_name,'-'*20, sep='')
    X, y = UnderSampling(df_temp, df['Bankrupt?'])
    cv_5_list = []
    for model_name, model in models.items():
        # if model_name != 'RF':
        #     continue
        print(model_name, end=' ')
        start_time = time.time()
        cv_5 = Analysis(X, y, model, option='cv')
        # mean_cv_5, mean_cv_10 = Analysis(X, y, model, option='cv')
        # print(f'5-fold : {mean_cv_5:.4f}\n10-fold : {mean_cv_10:.4f}\n')
        # cv_5_list.append(mean_cv_5)
        # cv_10_list.append(mean_cv_10)
        
        end_time = time.time()
        print(f"{end_time - start_time:.5f} sec")
        cv_5_list.append(cv_5)
            
        # plt.hist(cv_5, bins = 10, color = 'grey')
        # plt.xlim([0.60, 1])
        # plt.xlabel('F1 score')
        # plt.ylabel('Frequency')
        # title = f'{df_name} : {model_name}'
        # plt.title(title)
        # title = title.replace(' : ', '_')
        # plt.savefig(f'cv_5_{title}.png')
        # plt.show()
        # print(cv_5.min(), cv_5.max())
    
    cv_5_dict[df_name] = cv_5_list
df_cv_5 = pd.DataFrame(cv_5_dict, index = list(models.keys()))
#%%
for model_name in models.keys():
    sns.histplot(df_cv_5.loc[model_name,'Applying PCA'], bins = 10, label=model_name, alpha = .4, kde = True)
# for i, cv_5 in enumerate(cv_5_list):
#     df_name = list(dfs.keys())[i]
#     sns.histplot(cv_5, bins = 10, label=df_name, alpha = .4, kde = True)
sns.histplot(df_cv_5.loc['RF','Applying VIF'], bins = 10, alpha = .4, kde = True, color = 'r', label = 'RF with Applying VIF')
sns.histplot(df_cv_5.loc['RF','Applying FA'], bins = 10, alpha = .4, kde = True, color = 'g', label = 'RF with Applying FA')
sns.histplot(df_cv_5.loc['RF','Applying PCA'], bins = 10, alpha = .4, kde = True, color = 'b', label = 'RF with Applying PCA')
# df_cv_5.loc['RF','Applying PCA'].min()
plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim([0.60, 1])
plt.xlabel('F1 score')
plt.ylabel('Frequency')
plt.title('5-Fold Cross Validation')
plt.savefig('5_fold_cv_RF.png')
plt.show()

sns.histplot(df_cv_5.loc['GB','Applying VIF'], bins = 10, alpha = .4, kde = True, color = 'r', label = 'GB with Applying VIF')
sns.histplot(df_cv_5.loc['GB','Applying FA'], bins = 10, alpha = .4, kde = True, color = 'g', label = 'GB with Applying FA')
sns.histplot(df_cv_5.loc['GB','Applying PCA'], bins = 10, alpha = .4, kde = True, color = 'b', label = 'GB with Applying PCA')

plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim([0.60, 1])
plt.xlabel('F1 score')
plt.ylabel('Frequency')
plt.title('5-Fold Cross Validation')
plt.savefig('5_fold_cv_GB.png')
plt.show()

sns.histplot(df_cv_5.loc['XGB','Applying VIF'], bins = 10, alpha = .4, kde = True, color = 'r', label = 'XGB with Applying VIF')
sns.histplot(df_cv_5.loc['XGB','Applying FA'], bins = 10, alpha = .4, kde = True, color = 'g', label = 'XGB with Applying FA')
sns.histplot(df_cv_5.loc['XGB','Applying PCA'], bins = 10, alpha = .4, kde = True, color = 'b', label = 'XGB with Applying PCA')

plt.legend(loc = 'upper left', fontsize = 8)
plt.xlim([0.60, 1])
plt.xlabel('F1 score')
plt.ylabel('Frequency')
plt.title('5-Fold Cross Validation')
plt.savefig('5_fold_cv_XGB.png')
plt.show()

# cv_5 = pd.DataFrame([cv_5_list[i:i + len(models)] for i in range(0, len(cv_5_list), len(models))], 
#                     index = list(dfs.keys()), 
#                     columns = list(models.keys()))
# # cv_5.to_html(link+'\cv_5.html')

# cv_10 = pd.DataFrame([cv_10_list[i:i + len(models)] for i in range(0, len(cv_10_list), len(models))], 
#                      index = list(dfs.keys()), 
#                      columns = list(models.keys()))
# # cv_10.to_html(link+'\cv_10.html')
#%% 평균값 높은거 찾기
df_cv_5
for col in df_cv_5.columns:
    for model_name in models.keys():
        if df_cv_5.loc[model_name, col].mean() > .84:
            print(col, model_name, np.round(df_cv_5.loc[model_name, col].mean(), 5))

#%% cv_5, errorbar
lst_model = models.keys()
# lst_model = ['RF','GB','XGB']

for df_name in dfs.keys():
    score_f1_macro = []
    score_f1_macro_err = [[],[]]
    for model_name in lst_model:
        score_f1_macro.append(cv_5[model_name][df_name][0])
        score_f1_macro_err[0].append(cv_5[model_name][df_name][1])
        score_f1_macro_err[1].append(cv_5[model_name][df_name][2])
    plt.errorbar(lst_model, score_f1_macro, score_f1_macro_err, 
                 fmt='o', 
                 color='k',
                 capsize = 6
                 )
    plt.ylabel('f1 Score')
    plt.ylim([0.63,0.920])
    plt.grid(axis='y', linestyle='--')
    plt.title(df_name)
    plt.show()
    plt.savefig(f'{df_name}.png')

#%% cv_10, errorbar
# for df_name in dfs.keys():
#     score_f1_macro = []
#     score_f1_macro_err = [[],[]]
#     for model_name in lst_model:
#         score_f1_macro.append(cv_10[model_name][df_name][0])
#         score_f1_macro_err[0].append(cv_10[model_name][df_name][1])
#         score_f1_macro_err[1].append(cv_10[model_name][df_name][2])
#     plt.errorbar(lst_model, score_f1_macro, score_f1_macro_err, 
#                  fmt='o', 
#                  color='k',
#                  capsize = 6
#                  )
#     plt.ylabel('f1 Score')
#     plt.ylim([0.5,1.00])
#     plt.grid(axis='y', linestyle='--')
#     plt.title(df_name)
#     plt.show()


#%% roc curve
# lst_model = ['RF','GB','XGB']
lst_model = models.keys()
from sklearn.metrics import roc_curve, roc_auc_score

for df_name in dfs.keys():
    print(df_name)
    df_temp = dfs[df_name]
    for model_name in lst_model:
        model = models[model_name]
        print(model_name)
        y_test, y_pred = Analysis(df_temp, df['Bankrupt?'], model)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, '--')
    plt.title(df_name)
    plt.legend(lst_model, ncol=3, loc='lower right')
    plt.plot([0,1],[0,1], color='#000000')
    # plt.savefig(f'{link}\{df_name}_roc.png')
    plt.show()

#%% 20241213
from sklearn.metrics import roc_curve, roc_auc_score
lst_model = models.keys()
for df_name in dfs.keys():
    print(df_name)
    df_temp = dfs[df_name]
    plt.figure(figsize=(10, 8))
    for model_name in lst_model:
        model = models[model_name]
        X, y = UnderSampling(df_temp, df['Bankrupt?'])
        fprs, tprs, thresholds, auc_score = Analysis(X, y, model, option = 'roc')
        plt.plot(fprs, tprs, label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(df_name, fontsize = 20)
    
    plt.legend(ncol = 3, loc = 'lower right')
    plt.plot([0,1], [0,1], label = 'Threshold', color = 'k')
    # plt.savefig(f'roc curve {df_name}.png')
    plt.show()

#%% Feature Importance
# , 'Applying PCA'
A = []
for df_name in ['Applying VIF', 'Applying FA']:
    df_temp = dfs[df_name]
    for model_name in ['RF', 'GB', 'XGB']:
        model = models[model_name]
        
        X, y = UnderSampling(df_temp, df['Bankrupt?'])
        
        #Hold_Out
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)
        
        #Modeling
        model.fit(x_train, y_train)
        
        importances = model.feature_importances_
        indices = importances.argsort()
        
        # visualization
        # plt.figure(figsize=(10,20))
        # plt.barh([X.columns[i] for i in indices], importances[indices])
        # plt.xlabel("Feature Importance")
        # plt.title(f"{df_name} : {model_name} Feature Importances", fontsize = 15)
        # plt.savefig(f'{df_name}_{model_name} Feature Importances.png')
        # plt.show()
        A.append([X.columns[i] for i in indices][-10:])




K.to_csv('K.csv')

































