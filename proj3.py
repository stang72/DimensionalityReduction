import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.stats import kurtosis

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA,FastICA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

'''Load data'''
# Wine Data
wineData = pd.read_csv("data/wine/wine_out.csv",sep=',')
feature_name = wineData.columns[1:7]
wineMtx=wineData.values[:,1:8]

wineFeature=wineMtx[:,0:6]
wineLabel=wineMtx[:,-1]



# Credit Data
loanData = pd.read_csv("data/loan/credit_out.csv")
cleanup_nums = {True: 1, False: 0}
loanData.MISSING_DEBTINC=loanData.MISSING_DEBTINC.map(cleanup_nums)

feature_name = loanData.columns[1:7]
loanMtx=loanData.values[:,1:8].astype(str).astype(float)

loanFeature=loanMtx[:,0:6]
loanLabel=loanMtx[:,-1]

xTrain_total, xTest_total, yTrain_total, yTest_total = train_test_split(loanFeature,loanLabel,test_size=0.2,random_state=0)


'''Selection Criteria'''
def compute_aic_bic(kmeans,X):
    """
    Computes the AIC BIC metric for a given clusters

    Parameters:
    kmeans:  List of clustering object from scikit learn
    X     :  multidimension np array of data points

    Returns:
    AIC BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    ln_likelihood = np.sum([n[i] * np.log(n[i]) -
           n[i] * np.log(N) -
         ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
         ((n[i] - 1) * d/ 2) for i in range(m)]) 

    AIC = 2*m  - 2* ln_likelihood    
    BIC =  m * np.log(N) * (d+1) - 2* ln_likelihood

    return AIC, BIC


'''K-means clustering'''
def best_km_cluster(X, max_cluster = 5, title = None):

    ks = range(1,max_cluster+1)
    
    kms = [KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]
    
    clst, aic, bic = [],[],[]
    for i in range(len(kms)):
        temp = compute_aic_bic(kms[i],X) 
        clst.append(ks[i])
        aic.append(temp[0])
        bic.append(temp[1])
        
    df_cluster= pd.DataFrame(data= {'cluster': clst, 'aic': aic, 'bic': bic}, columns=['cluster', 'aic', 'bic'])
    
    plt.plot(df_cluster['cluster'],df_cluster['aic'], '-o',label = 'AIC')
    plt.plot(df_cluster['cluster'],df_cluster['bic'], '-o',label = 'BIC')
    plt.grid()
    plt.legend()
    if title == None:
        plt.title('K Means')
    else:
        plt.title('K Means:' + title)  
    plt.show()    
    
    return df_cluster


'''Expectation Maximization clustering'''
def best_em_cluster(X, max_cluster = None, title = None):
 
    ks = range(1,max_cluster)
    
    gmm = [GaussianMixture(n_components = i).fit(X) for i in ks]
    
    clst, aic, bic = [],[],[]
    for i in range(len(gmm)):
        clst.append(ks[i])
        aic.append(gmm[i].aic(X))
        bic.append(gmm[i].bic(X))
        
    df_cluster= pd.DataFrame(data= {'cluster': clst, 'aic': aic, 'bic': bic}, columns=['cluster', 'aic', 'bic'])
    
    plt.plot(df_cluster['cluster'],df_cluster['aic'], '-o',label = 'AIC')
    plt.plot(df_cluster['cluster'],df_cluster['bic'], '-o',label = 'BIC')
    plt.grid()
    plt.legend()
    if title == None:
        plt.title('EM')
    else:
        plt.title('EM:' + title)        
    plt.show()    
  
    return df_cluster 


'''Dim reduction PCA'''
def best_k_PCA(X, expected_explained_variance = 0.99):
	explained_variance = 0.0
	i = 1
	largest = X.shape[1]
	while (explained_variance<expected_explained_variance):
		pca = PCA(n_components=i)
		X_r = pca.fit(X).transform(X)
		explained_variance = np.cumsum(pca.explained_variance_ratio_)
		explained_variance = explained_variance[-1]
		if(explained_variance > expected_explained_variance):
			return (X_r,i,pca.explained_variance_ratio_)
		else:
			if(i<=largest-1):
				i+=1
			else:
				return (0,0,0)


def pltPCA_explainedVariance(X,dataset_name):
	largest = X.shape[1]
	pca = PCA(n_components=largest)
	X_r = pca.fit(X).transform(X)
	print (pca.explained_variance_ratio_)
	print (np.array(range(1,largest+1)))
	df_pc = pd.DataFrame(data = {'PC_number': range(1,largest+1), 'explained_variance': pca.explained_variance_ratio_}, columns = ['PC_number','explained_variance'])
	plt.plot(df_pc['PC_number'],df_pc['explained_variance'], '-o',label = 'explained_variance')
	plt.grid()
	plt.legend()
	plt.title('PCA explained variance'+dataset_name)
	plt.show()
	return pca.explained_variance_ratio_


def dimReductPCA(X,comp = 2):
	pca = PCA(n_components=comp)
	X_r = pca.fit(X).transform(X)
	return X_r


'''Dim reduction ICA'''
def dimReductICA(X,comp = 3):
	ica = FastICA(n_components=comp)
	X_r = ica.fit(X).transform(X)
	df_r = pd.DataFrame(X_r)
	print (kurtosis(df_r,axis = None,fisher = False))
	return X_r


'''Dim reduction RCA'''
def find_best_state_RCA(X,comp = 2, n_state = 20):
	reconstuction_error = []
	for i in range(n_state):
		rca = GaussianRandomProjection(n_components=comp,random_state=i)
		X_r = rca.fit(X).transform(X)
		X_inverse = np.matmul(X_r, rca.components_)
		similarity = cosine_similarity(X_inverse,X)[0][0]
		reconstuction_error.append(similarity)
	return reconstuction_error


'''Dim reduction reconstruction error'''
def reconstruction_similarity(X,method_name,comp = 2):
	if method_name == 'PCA':
		pca = PCA(n_components=comp)
		X_r = pca.fit(X).transform(X)
		X_inverse = pca.inverse_transform(X_r)
	if method_name == 'ICA':
		ica = FastICA(n_components=comp)
		X_r = ica.fit(X).transform(X)
		X_inverse = ica.inverse_transform(X_r)
	if method_name == 'RCA':
		rca = GaussianRandomProjection(n_components=comp)
		X_r = rca.fit(X).transform(X)
		X_inverse = np.matmul(X_r, rca.components_)

	similarity = cosine_similarity(X_inverse,X)[0][0]
	
	return similarity

def neuralNetwork(features, labels, trainingSize,level1=100,level2=50):
    xTrain, xTest, yTrain, yTest = train_test_split(features,labels,test_size=1-trainingSize,random_state=0)
    print ()
    # Data normalization
    scaler = preprocessing.StandardScaler().fit(xTrain)
    xTrain_scaled = scaler.transform(xTrain)
    xTest_scaled = scaler.transform(xTest)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(level1, level2,), random_state=1)
    clf.fit(xTrain_scaled, yTrain)
    prediction = clf.predict(xTest_scaled)
    score_f1 = f1_score(yTest, prediction)
    #print ("Cohen: "+str(score_f1))
    score_ac = clf.score(xTest_scaled, yTest)
    #print ("Accuracy: "+str(score_ac))
    # Cross validation
    # xCV = np.append(xTrain,xTrain_total,axis = 0)
    # yCV = np.append(yTrain,yTrain_total)
    # score_cv_ac = cross_val_score(clf, xCV, yCV).mean()
    # print ("CV Accuracy: "+str(score_cv_ac))
    score_cv_ac = clf.score(X=xTrain_scaled,y=yTrain)
    print (trainingSize+0.0001,score_ac,score_cv_ac,score_f1)

    return [trainingSize+0.0001,score_ac,score_cv_ac,score_f1]


if __name__ == "__main__":
	# k-mean clustering
#	best_km_cluster(wineFeature,30,"Wine")
#	best_km_cluster(loanFeature,30,"Loan")
	# EM clustering
#	best_em_cluster(wineFeature,30,"Wine")
#	best_em_cluster(loanFeature,30,"Loan")
	# Dimensionality reduction PCA
#	dimReduct_wine = best_k_PCA(wineFeature)
#	dimReduct_loan = best_k_PCA(loanFeature)
#	pltPCA_explainedVariance(wineFeature,' Wine')
#	pltPCA_explainedVariance(loanFeature,' Loan')
#	for i in range(1,7):
#		dimReductICA(wineFeature,' Wine',i)
#	for i in range(1,7):
#		dimReductICA(loanFeature,' Wine',i)
#	#dimReductICA(wineFeature,' Wine',6)
#	print ("PCA")
#	print (reconstruction_similarity(wineFeature,'PCA'))
#	print (reconstruction_similarity(loanFeature,'PCA'))
#	print ("ICA")
#	print (reconstruction_similarity(wineFeature,'ICA',3))
#	print (reconstruction_similarity(loanFeature,'ICA',3))
#	print ("RCA")
#	print (find_best_state_RCA(wineFeature,5,20))
#	print (find_best_state_RCA(loanFeature,5,20))
	wine_pca = dimReductPCA(wineFeature)
	credit_pca = dimReductPCA(loanFeature)
#	best_km_cluster(wine_pca,30,"Wine")
#	best_km_cluster(credit_pca,30,"Loan")
#	best_em_cluster(wine_pca,30,"Wine")
#	best_em_cluster(credit_pca,30,"Loan")
	wine_ica = dimReductICA(wineFeature)
	credit_ica = dimReductICA(loanFeature)
#	best_km_cluster(wine_ica,30,"Wine")
#	best_km_cluster(credit_ica,30,"Loan")
#	best_em_cluster(wine_ica,30,"Wine")
#	best_em_cluster(credit_ica,30,"Loan")
#	print (neuralNetwork(credit_pca,loanLabel,0.8))
	training_size = np.arange(1,11)/10-0.0001
	score_df = pd.DataFrame(columns=['Algorithm', 'Training_size', 'Testing_accuracy','Training_accuracy','F1_score'])
	for i in range(len(training_size)):
		score_array = ['NeuralNetwork']
		score_array = np.append(score_array,neuralNetwork(credit_ica,loanLabel,training_size[i]))
		score_df.loc[i+20]=score_array
	print ("done neuralNetwork")



