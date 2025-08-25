from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df=pd.read_csv("C:/Users/Omar/Downloads/spam.csv", encoding= "latin-1")
x=df["v2"]
y=df["v1"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
pipeline= Pipeline([("vect", CountVectorizer()),("teez", TfidfTransformer()),("nominal",MultinomialNB())])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(pipeline.score(x_test, y_test))