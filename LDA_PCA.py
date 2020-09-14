# %%
import data_plot
import matplotlib.figure as fig
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import FastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import manifold
from time import time
import pdb

digits=datasets.load_digits(n_class=6)

X=digits.data
y=digits.target
n_samples, n_features = X.shape
n_comp=2

Methods={
    'PCA' : PCA(n_components=n_comp),
    'LDA' : LinearDiscriminantAnalysis(n_components=n_comp),
    'MDS' : manifold.MDS(n_components=n_comp, max_iter=200, n_init=1),
    'tSNE' : manifold.TSNE(n_components=n_comp, init='pca', random_state=0),
    'ICA' : FastICA(n_components=n_comp,random_state=0)
    }



for i, (label,method) in enumerate(Methods.items()):
    t0=time()
    if label=='LDA':
        Y=method.fit_transform(X,y)
    else:      
        Y=method.fit_transform(X)
    t1=time()
    
    print("%s: %.2g sec" % (label, t1 - t0))

    #pdb.set_trace()
    data_plot.plot_digits_embedding(Y, y,label)
    