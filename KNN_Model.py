import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
data=pd.read_csv('C:/Users/Omar/Downloads/data.csv')
x=data.loc[:,["texture_mean"]]
y=data.loc[:,["fractal_dimension_worst"]]
M_data=data["diagnosis"] == "M"
B_data=data["diagnosis"] == "B"

plt.scatter(x[M_data],y[M_data],color="blue",label="Malignant(GOOD)")
plt.scatter(x[B_data],y[B_data],color="red",label="Benign(BAD)")
plt.title("Breast Cancer Diagnosis")
plt.legend()
plt.show()
#MACHINE LEARNING PART
data_clean=data
X=data_clean.iloc[:,2:29]
Y=data_clean.iloc[:,1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
print(knn.score(X_test, Y_test))
joblib.dump(knn, 'KNN_model.pkl')