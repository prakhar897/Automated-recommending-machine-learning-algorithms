from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import pandas as pd
import numpy as np

def Run_Algorithm_On_Dataset(dataset,y_axis):
	df = pd.read_csv(dataset)
	print(df.head())
	X = df.drop(columns=[y_axis])
	print(X.head())
	y = df[y_axis].values
	print(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	knn = KNeighborsClassifier(n_neighbors = 3)# Fit the classifier to the data
	knn.fit(X_train,y_train)
	y_pred = knn.predict(X_test)
	print(knn.predict(X_test)[0:5])
	print(knn.score(X_test, y_test))
	print(f1_score(y_test, y_pred, average="macro"))
	print(precision_score(y_test, y_pred, average="macro"))
	print(recall_score(y_test, y_pred, average="macro")) 
	
Run_Algorithm_On_Dataset("./Datasets/diabetes.csv","Outcome")

