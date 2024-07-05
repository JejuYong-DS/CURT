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
#!pip install factor_analyzer
#!pip install imblearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#%% Function
#값의 형태 확인
def unique_preprocess(df): 
    for i in df.columns:
        print(i, len(df[i].unique()))
        #Unique()가 1인 변수 제거
        if len(df[i].unique()) == 1:
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
    plt.figure(figsize=(12,17))
    sns.heatmap(df_FA_loadings,
                cmap="RdBu",
                annot=True,
                fmt='.2f',
                vmax=1, vmin=-1)
    plt.title(f'<{title}>')
    if title:
        plt.savefig(f'{link}\{title}.png', bbox_inches='tight')
    plt.show()

# Scree Plot
def scree_plot(x,y,title=None):
    plt.plot(x, y, 'o-',  color='black')
    plt.title('Scree Plot')
    plt.xlabel('n_feature')
    plt.ylabel('Variance Explained')
    plt.title(f'<{title}>')
    if title:
        plt.savefig(f'{link}\{title}.png')
    plt.show()    

#%% 1. raw data
#Call dataset
link = r'C:\Users\PC\Desktop\Korea_Univ\CDS_LAB\연구2\CURT\DB'
df = pd.read_csv(link+'\data.csv')
y = df['Bankrupt?']

#Unique Preprocess : 변수 1개 제거
df = unique_preprocess(df)

#중복된 열 제거
df = df.loc[:, ~df.T.duplicated()]

#missing values check : 결측값 없음
df.isnull().sum().sum()

#Descriptive Statistics
discribe_df = df.describe()
discribe_df.to_html(link+'\discribe_df.html')

#Scailing
scaler = StandardScaler()
df_raw = pd.DataFrame(scaler.fit_transform(df.iloc[:,1:]), columns = df.columns[1:])

#save file
df_raw.to_json(link+'\df_raw.json')

#Descriptive Statistics
discribe_df_raw = df_raw.describe()
discribe_df_raw.to_html(link+'\discribe_df_raw.html')

#%% 2. VIF
df_VIF = df_raw.copy()

vif = vif_calculate(df_VIF)
print(vif)
# vif.to_html(link+'\VIF.html')

# if VIF > 10, list-up
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
ev,v = fa.get_eigenvalues()
# print(ev)
ev_1 = [i for i in ev if i >= 1] # ev >= 1

#rotation을 None
df_FA_loadings, fa = factor_analysis(len(ev_1), df_FA, method=None)
FA_heatmap(df_FA_loadings, title='Rotation None')

#orthogonal rotation인 varimax 선택
df_FA_loadings, fa = factor_analysis(len(ev_1), df_FA, method='varimax')
FA_heatmap(df_FA_loadings, title='Rotation Varimax')

#분석 결과
fa.get_factor_variance() #누적 분산 0.451272
df_FA_variance = pd.DataFrame(fa.get_factor_variance(), index=['SS Loadings', 'Proportion Var', 'Cumulative Var'])

#save file
df_FA_variance.to_html(link+'\df_FA_variance.html')

#K = 0.3 이상의 적재량(절댓값)이 한번도 안나온 변수 제거
K = 0.3
loadings_under_list = []
for i in range(len(df_FA_loadings)):
    loadings_under = [abs(k) >= K for k in df_FA_loadings.iloc[i]]
    if sum(loadings_under) == 0:
        loadings_under_list.append(df_FA_loadings.index[i])

df_FA = df_FA.drop(loadings_under_list, axis = 1)
#save file
df_FA.to_json(link+'\df_FA.json')

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
scree_plot(PC_values, ve, title='Scree All')

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
df_PCA.to_json(link+'\df_PCA.json')

#Descriptive Statistics
discribe_df_PCA = df_PCA.describe()
discribe_df_PCA.to_html(link+'\discribe_df_PCA.html')

#%% Analysis
def Analysis(X,y, model, option=None):
    """
    option : 'evaluation', 'cv'
    """
    #UnderSampling
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=random_state)
    X, y = rus.fit_resample(X, y)
    
    #Hold_Out
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)
    
    #Modeling
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    if option == 'evaluation':
        #Odds Ratio
        from sklearn.metrics import confusion_matrix
        [[TP,FN], [FP,TN]] = confusion_matrix(y_test,y_pred)
        
        Odds_Ratio = (TP*TN)/(FN*FP)
        print('Confusion Matrix :\n', confusion_matrix(y_test,y_pred))
        print('Odds Ratio :', round(Odds_Ratio, 6))
        
        print('=' *53)
        
        #모델 평가 지표
        print('Train Data :', round(model.score(x_train, y_train), 6))
        print('Test Data :', round(model.score(x_test, y_test), 6))
        
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred))
        
        #AUC
        roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
        
        return round(model.score(x_train, y_train), 6), round(model.score(x_test, y_test), 6), roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    
    if option == 'cv':
        cv_5 = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        cv_10 = cross_val_score(model, X, y, cv=10, scoring='f1_macro')
        return np.mean(cv_5), np.mean(cv_10)
    
    return y_test, y_pred

#%% Model fitting
df_names = ['df_raw', 'df_VIF', 'df_FA', 'df_PCA']
model_names = ['Logistic', 'SVM', 'RF']

dfs = [df_raw, df_VIF, df_FA, df_PCA]
models = []

from sklearn.linear_model import LogisticRegression
models.append(LogisticRegression(random_state=random_state))

from sklearn import svm
models.append(svm.SVC(kernel='linear', C=1, random_state=random_state,probability=True))

from sklearn.ensemble import RandomForestClassifier
models.append(RandomForestClassifier(max_depth=5, random_state=random_state))

#%% evaluation
score_col_list = []
score_train_list = [] 
score_test_list = []
auc_list = []

for df_temp, df_name in zip(dfs, df_names):
    print(df_name)
    for model, model_name in zip(models, model_names):
        print(model_name)
        score_train, score_test, auc = Analysis(df_temp, df['Bankrupt?'], model, option='evaluation')

        score_col_list.append((df_name,model_name))
        score_train_list.append(score_train)
        score_test_list.append(score_test)
        auc_list.append(auc)

score_df = pd.DataFrame([score_train_list, score_test_list, auc_list], index = ['train_score', 'test_score', 'AUC'], columns = score_col_list).T
score_df = score_df.reindex(pd.MultiIndex.from_tuples(score_col_list))

score_df.to_html(link+'\score_df.html')

#%% cv
cv_5_list = []
cv_10_list = []
for df_temp, df_name in zip(dfs, df_names):
    print(df_name)
    for model, model_name in zip(models, model_names):
        print(model_name)
        mean_cv_5, mean_cv_10 = Analysis(df_temp, df['Bankrupt?'], model, option='cv')
        # print(f'5-fold : {mean_cv_5:.4f}\n10-fold : {mean_cv_10:.4f}\n')
        cv_5_list.append(mean_cv_5)
        cv_10_list.append(mean_cv_10)

cv_5 = pd.DataFrame([cv_5_list[i:i + len(models)] for i in range(0, len(cv_5_list), len(models))], index = df_names, columns=model_names)
cv_5.to_html(link+'\cv_5.html')

cv_10 = pd.DataFrame([cv_10_list[i:i + len(models)] for i in range(0, len(cv_10_list), len(models))], index = df_names, columns=model_names)
cv_10.to_html(link+'\cv_10.html')

#%% roc curve
for df_temp, df_name in zip(dfs, df_names):
    print(df_name)
    for model, model_name in zip(models, model_names):
        print(model_name)
        y_test, y_pred = Analysis(df_temp, df['Bankrupt?'], model)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, '--')
    plt.title(df_name)
    plt.legend(model_names)
    plt.plot([0,1],[0,1], color='#000000')
    plt.savefig(f'{link}\{df_name}_roc.png')
    plt.show()
    
#%% col explain
col_temp1 = list(df_raw.columns)
col_temp2 = ['이자 및 감가상각 전 이자 전 ROA(C)', '세후 이자 전 ROA(A) 및 %', '이자 및 세후 감가상각 전 ROA(B)', '운영 총이익율', '실현된 매출 총이익율', '운영 이윤률', '세전 순 이자율', '세후 순 이자율', '비업종 수입 및 지출/수익', '연속 이자율 (세후)', '운영 비용률', '연구 및 개발 비용률', '현금 흐름율', '이자 부채 이자율', '세율 (A)', '주당 순 자산 가치 (B)', '주당 순 자산 가치 (A)', '주당 순 자산 가치 (C)', '최근 네 분기간 지속 EPS', '주당 현금 흐름', '주당 매출액 (Yuan ¥)', '주당 운영 이익 (Yuan ¥)', '주당 세전 순 이익 (Yuan ¥)', '실현된 매출 총이익 성장률', '운영 이익 성장률', '세후 순 이익 성장률', '정기 순 이익 성장률', '연속 순 이익 성장률', '총자산 성장률', '순 자산 성장률', '총자산 수익 성장률 비율', '현금 재투자 %', '유동비율', '당좌비율', '이자 지출 비율', '총부채/총순자산', '부채 비율 %', '순자산/자산', '장기 자금 적합성 비율 (A)', '차입 종속성', '부채 보증/순 자산', '운영 이익/납입 자본', '세전 순 이익/납입 자본', '재고 및 매출채권/순 가치', '총 자산 회전율', '매출채권 회전율', '평균 수금일', '재고 회전율 (회수)', '고정 자산 회전 빈도', '순 자산 회전율 (회수)', '인당 매출', '인당 운영 이익', '인당 할당율', '운영 자본/총 자산', '당좌 자산/총 자산', '유동자산/총 자산', '현금/총 자산', '당좌 자산/유동 부채', '현금/유동 부채', '유동 부채/자산', '운영 자금/부채', '재고/운영 자본', '재고/유동 부채', '유동 부채/부채', '운영 자본/자본', '유동 부채/자본', '장기 부채/유동 자산', '유지 이익/총 자산', '총 수입/총 비용', '총 비용/자산', '유동 자산 회전율', '당좌 자산 회전율', '운영 자본 회전율', '현금 회전율', '매출에 대한 현금 흐름', '고정 자산/자산', '자본/장기 부채', '총 자산에 대한 현금 흐름', '부채에 대한 현금 흐름', 'CFO/자산', '자본에 대한 현금 흐름', '유동 부채/유동 자산', '부채-자산 플래그', '총 자산에 대한 순 이익', '총 자산에 대한 GNP 가격', '신용 없는 기간', '매출에 대한 총이익', '주주 자본에 대한 순 이익', '부채에 대한 자본', '금융 레버리지 정도 (DFL)', '이자 지급 보호 비율 (이자비용/EBIT)', '자본에 대한 부채']

col = pd.DataFrame([col_temp1, col_temp2]).T
col.to_html(link+'\col_explain.html')

#%%
col_dict = dict(zip(col_temp1, col_temp2))

