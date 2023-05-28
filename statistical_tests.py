import numpy as np
from scipy.stats import shapiro, normaltest, anderson, pearsonr, spearmanr, kendalltau, ttest_ind, friedmanchisquare, kruskal, wilcoxon, mannwhitneyu, f_oneway, ttest_rel
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import matplotlib as plt
import seaborn as sns
from scipy.stats import norm, kstest, shapiro, anderson, friedmanchisquare
import scikit_posthocs as sp


df = pd.read_csv('output/means_stations_def.csv')
df = df.filter(['dataset', 'mae', 'mae_2', 'mae_3', 'mae_stacked'], axis=1)
df_og = df

df_mae = df.filter(['dataset', 'mae'], axis=1)
df_mae_2 = df.filter(['dataset', 'mae_2'], axis=1)
df_mae_3 = df.filter(['dataset', 'mae_3'], axis=1)
df_mae_stacked = df.filter(['dataset', 'mae_stacked'], axis=1)

X_mae = df_mae['mae'].values
X_mae_2 = df_mae_2['mae_2'].values
X_mae_3 = df_mae_3['mae_3'].values
X_mae_stacked = df_mae_stacked['mae_stacked'].values

measure1 = df_og['mae'].values
measure2 = df_og['mae_2'].values
measure3 = df_og['mae_3'].values
measure4 = df_og['mae_stacked'].values

n_array = np.array([measure1, measure2, measure3, measure4])
n_array = np.transpose(n_array)

print('\n================================================================================================================================')
print('Siegel Friedman Posthoc Tests: ', sp.posthoc_siegel_friedman(n_array, p_adjust='holm'))
#print('Nemenyi Friedman Posthoc Tests: ', sp.posthoc_nemenyi_friedman(n_testing))
# print('Conover Posthoc Tests: ', sp.posthoc_mannwhitney(n_array, p_adjust='holm'))
# print('Dunn Posthoc Tests: ', sp.posthoc_wilcoxon(n_array, p_adjust='holm'))
# print('Nemenyi Posthoc Tests: ', sp.posthoc_nemenyi(n_array))
# print('Anderson Posthoc Tests: ', sp.posthoc_anderson(n_array, p_adjust='holm'))
# print('Ttest Posthoc Tests: ', sp.posthoc_ttest(n_array, p_adjust='holm'))
# print('Conover Friedman Posthoc Tests: ', sp.posthoc_conover_friedman(n_array, p_adjust='holm'))
print('================================================================================================================================')
