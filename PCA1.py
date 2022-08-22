
"""
"""
################# PCA(PRINCIPAL COMPONENT ANALYSIS) ############

"""Perform hierarchical and K-means clustering on the dataset. After that, 
perform PCA on the dataset and extract the first 3 principal components and 
make a new dataset with these 3 principal components as the columns. 
Now, on this new dataset, perform hierarchical and K-means clustering. 
Compare the results of clustering on the original dataset and clustering 
on the principal components dataset (use the scree plot technique to obtain 
the optimum number of clusters in K-means clustering and check if you’re getting
 similar results with and without PCA)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


wine=pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\Assignments\Assignment Datasets\wine.csv")
wine.info()#dtypes: float64(11), int64(3) and memory usage: 19.6 KB
wine### data preprocesing  ##############
wine.isna().sum()# no null values
wine.duplicated().sum()# no duplicate values
wine.columns
#finding outliers
sns.boxplot(wine["Type"])
sns.boxplot(wine["Alcohol"])
sns.boxplot(wine["Malic"])#outliers are present
sns.boxplot(wine["Ash"])#outliers are present
sns.boxplot(wine["Alcalinity"])#outliers are present
sns.boxplot(wine["Magnesium"])#outliers are present
sns.boxplot(wine["Phenols"])
sns.boxplot(wine["Flavanoids"])
sns.boxplot(wine["Nonflavanoids"])
sns.boxplot(wine["Proanthocyanins"])#outliers are present
sns.boxplot(wine["Color"])#outliers are present
sns.boxplot(wine["Hue"])#outliers are present
sns.boxplot(wine["Dilution"])
sns.boxplot(wine["Proline"])
#outliers are present in 7 columns

# Detection of outliers (find limits based on IQR)
IQR = wine['Malic'].quantile(0.75) - wine['Malic'].quantile(0.25)
lower_limit = wine['Malic'].quantile(0.25) - (IQR * 1.5)
upper_limit = wine['Malic'].quantile(0.75) + (IQR * 1.5)


# ####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Malic'])

df_t = winsor.fit_transform(wine[['Malic']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.Malic) # no outliers in Malic data



# Detection of outliers (find limits based on IQR)
IQR = wine['Ash'].quantile(0.75) - wine['Ash'].quantile(0.25)
lower_limit = wine['Ash'].quantile(0.25) - (IQR * 1.5)
upper_limit = wine['Ash'].quantile(0.75) + (IQR * 1.5)


# ####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Ash'])

df_t = winsor.fit_transform(wine[['Ash']])

# lets see boxplot
sns.boxplot(df_t.Ash) #no outliers in Ash



# Detection of outliers (find limits based on IQR)
IQR = wine['Alcalinity'].quantile(0.75) - wine['Alcalinity'].quantile(0.25)
lower_limit = wine['Alcalinity'].quantile(0.25) - (IQR * 1.5)
upper_limit = wine['Alcalinity'].quantile(0.75) + (IQR * 1.5)


# ####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Alcalinity'])

df_t = winsor.fit_transform(wine[['Alcalinity']])

# lets see boxplot
sns.boxplot(df_t.Alcalinity) #no outliers in Alcalinity


# Detection of outliers (find limits based on IQR)
IQR = wine['Magnesium'].quantile(0.75) - wine['Magnesium'].quantile(0.25)
lower_limit = wine['Magnesium'].quantile(0.25) - (IQR * 1.5)
upper_limit = wine['Magnesium'].quantile(0.75) + (IQR * 1.5)


# ####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Magnesium'])

df_t = winsor.fit_transform(wine[['Magnesium']])

# lets see boxplot
sns.boxplot(df_t.Magnesium)#  no outliers in Magnesium


# Detection of outliers (find limits based on IQR)
IQR = wine['Proanthocyanins'].quantile(0.75) - wine['Proanthocyanins'].quantile(0.25)
lower_limit = wine['Proanthocyanins'].quantile(0.25) - (IQR * 1.5)
upper_limit = wine['Proanthocyanins'].quantile(0.75) + (IQR * 1.5)

# ####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Proanthocyanins'])

df_t = winsor.fit_transform(wine[['Proanthocyanins']])

# lets see boxplot
sns.boxplot(df_t.Proanthocyanins) # no outliers in Proanthocyanins


# Detection of outliers (find limits based on IQR)
IQR = wine['Color'].quantile(0.75) - wine['Color'].quantile(0.25)
lower_limit = wine['Color'].quantile(0.25) - (IQR * 1.5)
upper_limit = wine['Color'].quantile(0.75) + (IQR * 1.5)


# ####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Color'])

df_t = winsor.fit_transform(wine[['Color']])

# lets see boxplot
sns.boxplot(df_t.Color)


# Detection of outliers (find limits based on IQR)
IQR = wine['Hue'].quantile(0.75) - wine['Hue'].quantile(0.25)
lower_limit = wine['Hue'].quantile(0.25) - (IQR * 1.5)
upper_limit = wine['Hue'].quantile(0.75) + (IQR * 1.5)

# ####  Winsorization ####
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Hue'])

df_t = winsor.fit_transform(wine[['Hue']])

# lets see boxplot
sns.boxplot(df_t.Hue)


#normalization
# converts range to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

wine_norm = norm_func(wine)
# now our data is scale free and unit free


############### Exploratory Data Analysis##################
# Measures of Central Tendency / First moment business decision
wine.mean() #high mean in Proline column
wine.median()#high median in proline column
#mean>median

from scipy import stats
stats.mode(wine)

# Measures of Dispersion / Second moment business decision
wine.var() # high variance in proline column
#high variance means more data spread 
wine.std() # high standard deviation in proline column

# Third moment business decision
wine.skew()

# Fourth moment business decision
wine.kurt()

############## data visuvalization #############
#histogram
plt.hist(wine,bins=9)#data is moderatly distributed normally


############### hierarchical clustering #############3

######### creating dendrogram  ######### 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

#computing the distance between data points using euclidean distance
a= linkage(wine_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(25, 10));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(a, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()
#by dendogram we can say that more distance of vertical lines means more distance b/w those clusters
#initialy it form 6 diff color culters.i.e 6 different groups

from sklearn.cluster import AgglomerativeClustering
# no of cluster depends on clint's requirements as it is subjective.
#we can change that as per business requirements 
#if we draw a horizantal atdistance 1.5 that line joint 6 horizantal lines
#so we can take no of cluster as 6

a_complete = AgglomerativeClustering(n_clusters = 6, linkage = 'complete', affinity = "euclidean").fit(wine_norm) 
a_complete.labels_ #it gives array of 6clsters 
cluster_labels = pd.Series(a_complete.labels_)

wine['clust'] = cluster_labels # creating a new column and assigning it to new column 
wine.insert(0,"clust",wine.pop("clust"))#it give 0 indexing to clust column

wine1 = wine.iloc[:, 0:15]
wine1.head()

# Aggregate mean of each cluster
a1=wine.iloc[:, :].groupby(wine.clust).mean()

#clust2 gp contains type1 wine.it contain high proline and less alcalinity  
#clust3 gp contains high magnesium ,phenols,flavanoids

#################   k-means #################
from sklearn.cluster import	KMeans 

###### scree plot or elbow curve ############
TWSS = [] #creating empty list for total within sum of sqr
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_norm)
    TWSS.append(kmeans.inertia_)#inertia is inbuild fn which gives twss
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
#from 2 to 3 it covered more data change and 3 to 4 little lethan 2 to 3
#so we can consider no of k mean clusters as 3

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(wine_norm)

model.labels_ # getting the labels of clusters assigned to each row 
k=pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine['clust']=k # creating a  new column and assigning it to new column 

wine.head()

wine = wine.iloc[:,:]
wine.head()

x=wine.iloc[:, :].groupby(wine.clust).mean()

#clust0 has high proline and magnesium
#clust2 has more alcalinity and malic and less flavanoids than other 2 clusters 

###################### PCA ####################

pca = PCA(n_components = 6)
pca_values = pca.fit_transform(wine_norm)

# The amount of variance that each PCA  
var = pca.explained_variance_ratio_
var # variance of each column #array([0.47611032, 0.1979128 , 0.06192905])

# PCA weights
pca.components_
pca.components_[0] #pc1 values

# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
#here we r taking var & rounding it upto 4 decimals and multiply with 100 for %
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns='pc0','pc1','pc2','pc3','pc4','pc5'
final_pca = pd.concat([wine.Type, pca_data.iloc[:, 0:3]], axis = 1)

# Scatter diagram
plot = final_pca.plot(x='pc0', y='pc1', kind='scatter',figsize=(12,8))

#now new dataset  is wine_pca
#let's perform hierarchical and K-means clustering on wine_pca


############### hierarchical clustering #############3

######### creating dendrogram  ######### 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

#computing the distance between data points using euclidean distance
b= linkage(final_pca, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(25, 10));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(b, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size =4 # font size for the x axis labels
)
plt.show()
#we are able to see clear visualization after pca

from sklearn.cluster import AgglomerativeClustering

b_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(wine_norm) 
b_complete.labels_  
cluster_labels = pd.Series(b_complete.labels_)

final_pca['clust'] = cluster_labels # creating a new column and assigning it to new column 
final_pca.insert(0,"clust",final_pca.pop("clust"))#it give 0 indexing to clust column

final1_pca = final_pca.iloc[:, 0:4]
final1_pca.head()

# Aggregate mean of each cluster
b1=final_pca.iloc[:, :].groupby(final_pca.clust).mean()
#clust1 of pc0 has more mean
#clust1 has type3 wine.it has more mean so it is prefered wine

#################   k-means #################
from sklearn.cluster import	KMeans 

###### scree plot or elbow curve ############
TWSS = [] #creating empty list for total within sum of sqr
k1 = list(range(2, 9))

for i in k1:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final_pca)
    TWSS.append(kmeans.inertia_)#inertia is inbuild fn which gives twss
    
TWSS
# Scree plot 
plt.plot(k1, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
#from 2 to 3 it covers max datapoints
#when compare to before pca k-means after pca k-meanscovers max datapoints
#at 2 to 3

# we can take no of cluster=3 from the above twss values
model1 = KMeans(n_clusters = 3)
model1.fit(final_pca)

model1.labels_ # getting the labels of clusters assigned to each row 
k1=pd.Series(model1.labels_)  # converting numpy array into pandas series object 
final_pca['clust']=k1 # creating a  new column and assigning it to new column 
final_pca.head()

final_pca = final_pca.iloc[:,:]
final_pca.head()

y=final_pca.iloc[:, :].groupby(final_pca.clust).mean()
y
 
#clust2 has more mean and it has type3 wine so it is most preffered one  
#we get similar results
#but pca is not as readable and interpretable because we lose some features

