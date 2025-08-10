# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
data= penguins_df.iloc[:,:4]
Stander= StandardScaler()
str_data=Stander.fit_transform(data)
model=KMeans(n_clusters=4, random_state=42)
model.fit(str_data)
penguins_df['labels']=model.labels_
plt.scatter(penguins_df['labels'], penguins_df['culmen_length_mm'], c=model.labels_)
plt.xlabel('Cluster')
plt.ylabel('culmen_length_mm')
plt.title(f'K-means Clustering (K={4})')
plt.show()