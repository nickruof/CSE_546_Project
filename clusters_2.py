import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os as os
import itertools

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_validate
from pandas.plotting import parallel_coordinates
from scipy.spatial.distance import cdist

from mpl_toolkits.mplot3d import Axes3D
sns.set()

plots_dir = "/Users/nickruof/Documents/Graduate_Courses/CSE_546/Project/plots/"
p_dir = "/Users/nickruof/Documents/Graduate_Courses/CSE_546/Project/params/"
w_dir = "/Users/nickruof/Documents/Graduate_Courses/CSE_546/Project/waves/"

def load_large_frame(params_dir = p_dir,waves_dir = w_dir):
    params_columns = ["channel","isEnr", "isGood","tOffset", "dtPulserCard","trapENMSample",
    "trapENFCal", "avse", "nlcblrwfSlope", "dcr99","trapETailMin", "kvorrT",
    "d2wfnoiseTagNorm", "dtmu_s", "waveS5", "bandMax", "den10", "den90", 
    "fitMu", "fitSlo", "pol0", "pol1", "pol2", "pol3","riseNoise","latAF",
    "wfStd"]
    p_files = os.listdir(p_dir)
    w_files = os.listdir(w_dir)
    large_params = pd.read_csv(p_dir+p_files[0], sep=" ", header=None)
    large_params.columns = params_columns
    large_waves = pd.read_csv(w_dir+w_files[0], sep=" ", header=None)
    print("Frame 0 Loaded!")
    for i in range(1,len(p_files)):
        params_data = pd.read_csv(params_dir+p_files[i],sep=" ", header=None)
        waves_data = pd.read_csv(waves_dir+w_files[i],sep=" ", header=None)
        params_data.columns = params_columns
        large_params = pd.concat([large_params,params_data],axis=0,ignore_index=True)
        large_waves = pd.concat([large_waves,waves_data],axis=0,ignore_index=True)
        print("Frame "+str(i)+ " Loaded!")
    return (large_params,large_waves)

def cluster_train(features,n_clusters=2,method="kmeans"):
    if(method == "kmeans"):
        kmeans = KMeans(n_clusters=n_clusters,n_init=10).fit(features)
        return kmeans
    elif(method == "gauss_mix"):
        gauss_mix = GaussianMixture(n_components=n_clusters).fit(features)
        return gauss_mix
    elif(method == "dbscan"):
        db = DBSCAN(eps=0.3, min_samples=10).fit(features)
        y = db.labels_
        clusters = len(set(y)) - (1 if -1 in y else 0)
        return (db, clusters)
    else:
        print("Model type not recognized!")

def plot_distortion_kmeans(features,name):
    distortions = []
    K = range(1,20)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(features)
        kmeanModel.fit(features)
        distortions.append(sum(np.min(cdist(features, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / features.shape[0])
    plt.figure()
    plt.plot(K,distortions)
    plt.xlabel("K Clusters")
    plt.ylabel("Distortion")
    plt.tight_layout()
    plt.savefig(plots_dir+"kmeans_distortions/dis_"+name+".png")

def plot_distortion_gauss(features,name):
    distortions = []
    K = range(1,20)
    for k in K:
        GaussModel = GaussianMixture(n_components=k).fit(features)
        GaussModel.fit(features)
        distortions.append(sum(np.min(cdist(features, GaussModel.means_, 'euclidean'), axis=1)) / features.shape[0])
    plt.figure()
    plt.plot(K,distortions)
    plt.xlabel("N Components")
    plt.ylabel("Distortion")
    plt.tight_layout()
    plt.savefig(plots_dir+"gauss_distortions/dis_"+name+".png")


def plot_silhouette(model,n_clusters,features,name):
    directory = ""
    y = None
    if(model == "kmeans"):
        kmeans = KMeans(n_clusters=n_clusters).fit(features)
        y = kmeans.labels_
        silhouette_vals = silhouette_samples(features,kmeans.labels_,metric="euclidean")
        directory = "kmeans_silhouette/"
    elif(model == "gauss_mix"):
        gauss_mix = GaussianMixture(n_components=n_clusters).fit(features)
        y = gauss_mix.predict(features)
        silhouette_vals = silhouette_samples(features,gauss_mix.predict(features),metric="euclidean")
        directory = "gauss_silhouette/"
    else:
        print("Model type not recognized!")
        return
    colors = ["red","blue","green","purple"]
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    plt.figure()
    for i, c in enumerate(range(0,n_clusters)):
        print(i)
        c_silhouette_vals = silhouette_vals[y == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = colors[i] #matplotlib.cm.jet(i / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals,height=1.0,edgecolor="none",color=color)
        yticks.append((y_ax_lower+y_ax_upper)/2)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,color="red",linestyle="--")
    plt.yticks(yticks,range(0,n_clusters))
    plt.ylabel("Clusters")
    plt.xlabel("Silhouette Coefficient")
    plt.savefig(plots_dir+directory+"sil_"+name+".png")

def plot_features_3D(features,labels,colors):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(features["u0"],features["u1"],features["u2"],edgecolors="w",c=labels,
           cmap=matplotlib.colors.ListedColormap(colors))
    ax.set_xlabel("1st Principal Mode")
    ax.set_ylabel("2nd Principal Mode")
    ax.set_zlabel("3rd Principal Mode")
    plt.show()


def plot_3D(params,features):
    params_columns = ["channel","isEnr", "isGood","tOffset", "dtPulserCard","trapENMSample",
    "trapENFCal", "avse", "nlcblrwfSlope", "dcr99","trapETailMin", "kvorrT",
    "d2wfnoiseTagNorm", "dtmu_s", "waveS5", "bandMax", "den10", "den90", 
    "fitMu", "fitSlo", "pol0", "pol1", "pol2", "pol3","riseNoise","latAF",
    "wfStd"]
    for variable in params_columns:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(features["u0"],features["u1"],features["u2"],edgecolors="w",c=params[str(variable)],
                   cmap=plt.get_cmap("hot"))
        ax.set_xlabel("1st Principal Mode")
        ax.set_ylabel("2nd Principal Mode")
        ax.set_zlabel("3rd Principal Mode")
        plt.legend([variable])
        plt.savefig(plots_dir + "colors/" + str(variable) + ".png")
        plt.close()


def categorize_waves(features,waveforms):
    n_clusters = max(features["gauss_labels"]) + 1
    for i in range(n_clusters):
        fig, axes = plt.subplots(5,5,figsize=(20,20), clear=True)
        clusters = waveforms.loc[features["gauss_labels"] == i,:]
        clusters = clusters.T
        clusters.columns = range(0,clusters.shape[1])
        sample_waveforms = clusters[np.random.randint(0,clusters.shape[1]-1,25)]
        sample_waveforms.columns = range(0,sample_waveforms.shape[1])
        for j in range(5):
            for k in range(5):
                axes[j,k].plot(sample_waveforms[j+5*k])
        plt.savefig(plots_dir + "waveforms/Cluster_" + str(i) + "_Waveforms.png")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()






#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

(params, waves) = load_large_frame()
norm_params = (params - params.mean())/params.std()
norm_params = norm_params.drop(["isEnr","isGood"],axis=1)
(u,s,v) = np.linalg.svd(norm_params,full_matrices=False)
features = pd.DataFrame(u.T[0:3].T,columns=["u0","u1","u2"])

kmeans = cluster_train(features,n_clusters=3,method="kmeans")
gauss_mix = cluster_train(features,n_clusters=3,method="gauss_mix")

#plot_silhouette("gauss_mix",n_clusters=4,features=features,name="gauss_final_2")


features["gauss_labels"] = gauss_mix.predict(features)
features["k_labels"] = kmeans.labels_
colors = ["red","blue","green"]
#categorize_waves(features,waves)


fig = plt.figure()
ax = Axes3D(fig)
p = ax.scatter(features["u0"],features["u1"],features["u2"],edgecolors="w",c=features["gauss_labels"],
           cmap=matplotlib.colors.ListedColormap(colors))
ax.set_xlabel("1st Principal Mode")
ax.set_ylabel("2nd Principal Mode")
ax.set_zlabel("3rd Principal Mode")
fig.colorbar(p, fraction=0.046, pad=0.04, ticks=[0,1,2])
plt.show()

#plot_3D(params,features)


SVM = SVC(gamma='auto')
forest = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
#cv_results_k = cross_validate(forest,features[["u0","u1","u2"]],features["k_labels"],cv=2)
#cv_results_gauss = cross_validate(forest,features[["u0","u1","u2"]],features["gauss_labels"],cv=2)
#cv_SVM_k = cross_validate(SVM,features[["u0","u1","u2"]],features["k_labels"],cv=2)
#cv_SVM_gauss = cross_validate(SVM,features[["u0","u1","u2"]],features["gauss_labels"],cv=2)
features = features.sample(frac=1).reset_index(drop=True)
(train, test) = np.split(features[["u0","u1","u2"]],[int(params.shape[0]*0.7)])
(train_labels, test_labels) = np.split(features["gauss_labels"],[int(params.shape[0]*0.7)])
SVM.fit(train,train_labels)
forest.fit(train,train_labels)


svm_confusion = confusion_matrix(test_labels,SVM.predict(test),labels=[0,1,2])
forest_confusion = confusion_matrix(test_labels,forest.predict(test),labels=[0,1,2])

plot_confusion_matrix(cm=svm_confusion,classes=colors,normalize=True)
plot_confusion_matrix(cm=forest_confusion,classes=colors,normalize=True)



print(cv_results_k["test_score"])
print(cv_results_gauss["test_score"])
print(cv_SVM_k["test_score"])
print(cv_SVM_gauss["test_score"])
