# %%
import data_plot
import matplotlib.figure as fig
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pdb

digits=datasets.load_digits(n_class=6)

X=digits.data
y=digits.target
n_samples, n_features = X.shape


Methods={
    'PCA' : PCA(n_components=2),
    'LDA' : LinearDiscriminantAnalysis(n_components=2),
    }
X_r=Methods['PCA'].fit(X).transform(X)
plt.figure()
data_plot.plot_digits_embedding(X_r,y)


X_r2=Methods['LDA'].fit(X,y).transform(X)
plt.figure()
data_plot.plot_digits_embedding(X_r2,y)

print('explained variance ratio, PCA:',Methods['PCA'].explained_variance_ratio_,'lDA: ',Methods['LDA'].explained_variance_ratio_)

