# %%
from scipy.io import savemat
import numpy as np


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

    
    
        
    
def dataset_fixCov(n=300, dim=2):
    np.random.seed(0)
    C=np.array([[0., -0.23],[0.84,0.23]])
    X=np.r_[np.matmul(np.random.randn(n,dim),C), np.matmul(np.random.randn(n,dim),C)+np.array([1,1])]
    y=np.r_[np.zeros(n), np.ones(n)]
    return X,y

def dataset_cov(n=300, dim=2):
    np.random.seed(0)
    C=np.array([[0., -1.],[2.5,.7]])*2
    X=np.r_[np.matmul(np.random.randn(n,dim),C), np.matmul(np.random.randn(n,dim),C)+np.array([1,4])]
    y=np.r_[np.zeros(n), np.ones(n)]
    return X,y


y_pred1=[];
X1,y1=dataset_fixCov()
ldaClassifer=LDA(solver="svd",store_covariance=True)
y_pred1.append(ldaClassifer.fit(X1,y1).predict(X1))
qdaClassifer=QDA(store_covariance=True)
y_pred1.append(qdaClassifer.fit(X1,y1).predict(X1))


y_pred2=[];
X2,y2=dataset_cov()
ldaClassifer=LDA(solver="svd",store_covariance=True)
y_pred2.append(ldaClassifer.fit(X2,y2).predict(X2))
qdaClassifer=QDA(store_covariance=True)
y_pred2.append(qdaClassifer.fit(X2,y2).predict(X2))
 
print('equal COV with LDA: ', np.sum(y1==y_pred1[0])/y1.size, ', with QDA', np.sum(y1==y_pred1[1])/y1.size,
'\ndifferent COV with LDA: ', np.sum(y2==y_pred2[0])/y2.size, ', with QDA', np.sum(y2==y_pred2[1])/y2.size) 
    

scipy.io.savemat('./LDA_QDA.mat',mdict={'X1':X1, 'y1':y1, 'X2':X2, 'y2':y2
                , 'y_pred1':y_pred1, 'y_pred2':y_pred2})
    