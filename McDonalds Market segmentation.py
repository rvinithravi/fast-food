#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C:\\Users\\PAGALAVAN SELVAM\\Downloads\\mcdonalds.csv")
df.shape


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df['yummy'].value_counts()


# In[8]:


df['VisitFrequency'].value_counts()


# In[9]:


df['Like'].value_counts()


# In[10]:


df['convenient'].value_counts()


# In[11]:


df['Age'].value_counts()


# In[12]:


MD_x = df.iloc[:, 0:11].values
MD_x = (MD_x == "Yes").astype(int)
col_means = np.round(np.mean(MD_x, axis=0), 2)

print(col_means)


# In[29]:


labels = ['Male','Female']
sizes = [df.query('Gender == "Male"').Gender.count(),df.query('Gender == "Female"').Gender.count()]
plt.pie(sizes,labels=labels)
plt.show()


# In[13]:


df['Like']= df['Like'].replace({'I hate it!-5': '-5','I love it!+5':'+5'})


# In[14]:


df.head(10)


# In[15]:


from sklearn.preprocessing import LabelEncoder
def labelling(x):
    df[x] = LabelEncoder().fit_transform(df[x])
    return df

cat = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap',
       'tasty', 'expensive', 'healthy', 'disgusting']

for i in cat:
    labelling(i)
df


# In[16]:


plt.rcParams['figure.figsize'] = (12,14)
df.hist()
plt.show()


# In[17]:


df1 = df.loc[:,cat]
df1


# In[18]:


x = df.loc[:,cat].values
x


# In[19]:


from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_data = preprocessing.scale(x)

pca = PCA(n_components=11)
pc = pca.fit_transform(x)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
pf = pd.DataFrame(data = pc, columns = names)
pf


# In[20]:


pca.explained_variance_ratio_


# In[21]:


np.cumsum(pca.explained_variance_ratio_)


# In[23]:


pca = PCA()
pca.fit(df1)
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC" + str(i) for i in range(1, num_pc+1)]
loadings_df = pd.DataFrame(loadings.T, columns=pc_list)
loadings_df['variable'] = df1.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


# In[30]:


plt.rcParams['figure.figsize'] = (20,15)
ax = sns.heatmap(loadings_df, annot=True)
plt.show()


# In[25]:


get_ipython().system('pip install bioinfokit')

from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_],show=True,dim=(10,5))


# In[26]:


pca_scores = PCA().fit_transform(x)


cluster.biplot(cscore=pca_scores, loadings=loadings, labels=df.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),
    var2=round(pca.explained_variance_ratio_[1]*100, 2),show=True,dim=(10,5))


# In[31]:


pip install yellowbrick


# In[33]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df1)
visualizer.show()


# In[35]:



kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df1)
df['cluster_num'] = kmeans.labels_ 
print (kmeans.labels_)
print (kmeans.inertia_) 
print(kmeans.n_iter_) 
print(kmeans.cluster_centers_)


# In[36]:


from collections import Counter
Counter(kmeans.labels_)


# In[37]:


sns.scatterplot(data=pf, x="pc1", y="pc2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()


# In[38]:


from statsmodels.graphics.mosaicplot import mosaic
from itertools import product

crosstab =pd.crosstab(df['cluster_num'],df['Like'])
#Reordering cols
crosstab = crosstab[['-5','-4','-3','-2','-1','0','+1','+2','+3','+4','+5']]
crosstab 


# In[40]:


crosstab_gender =pd.crosstab(df['cluster_num'],df['Gender'])
crosstab_gender


# In[41]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab_gender.stack())
plt.show()


# In[43]:


sns.boxplot(x="cluster_num", y="Age",data=df)


# In[45]:


df['VisitFrequency'] = LabelEncoder().fit_transform(df['VisitFrequency'])
visit = df.groupby('cluster_num')['VisitFrequency'].mean()
visit = visit.to_frame().reset_index()
visit


# In[46]:


df['Like'] = LabelEncoder().fit_transform(df['Like'])
Like = df.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
Like


# In[47]:


df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
Gender = df.groupby('cluster_num')['Gender'].mean()
Gender = Gender.to_frame().reset_index()
Gender


# In[48]:


segment = Gender.merge(Like, on='cluster_num', how='left').merge(visit, on='cluster_num', how='left')
segment


# In[49]:



plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Like", fontsize = 12) 
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




