import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from collections import defaultdict
import time
import ntpath
import json

from hyperparameters import AHP_features_matrix
from hyperparameters import features
from fuzzy_AHP import fuzzy_AHP
ntpath.basename("a/b/c")

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)



columns = defaultdict(list)

class Dataset:

	def __init__(self,dataset,y_axis):
		self.dataset = dataset
		self.y_axis = y_axis
		self.df = pd.read_csv(dataset)


	def makeCharacteristicsData(self):
		attributes = len(self.df.count())
		instanceCount = len(self.df)
		numericalAttributes = len(self.df.describe().count())
		self.df['full_count'] = self.df.apply(lambda x: x.count(), axis=1)
		missingCount =(instanceCount*attributes)-self.df['full_count'].sum()
		self.df = self.df.drop(columns=["full_count"])

		columns["dataset"].append("{}".format(path_leaf(self.dataset).split(".")[0]))
		columns["attributes"].append(attributes)
		columns["Instance_Count"].append(instanceCount)
		columns["Numerical_Attributes"].append(numericalAttributes)
		columns["Missing_Count"].append(missingCount)

		le = preprocessing.LabelEncoder()
		for col in self.df.columns:
			self.df[col] = self.df[col].astype(str)
			self.df[col] = le.fit_transform(self.df[col])
		self.X = self.df.drop(columns=[self.y_axis])
		self.y = self.df[self.y_axis].values
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
		self.myColumns = defaultdict(list)


	def runKNN(self):
		name = "K Nearest Neighbours"
		knn = KNeighborsClassifier(n_neighbors = 5)
		st = time.clock()
		knn.fit(self.X_train,self.y_train)
		trainingTime = round(time.clock()-st,6)
		st = time.clock()
		y_pred = knn.predict(self.X_test)
		testingTime = round(time.clock()-st,6)
		acs = accuracy_score(self.y_test, y_pred)
		fs = f1_score(self.y_test, y_pred, average="macro")
		ps = precision_score(self.y_test, y_pred, average="macro")
		rs = recall_score(self.y_test, y_pred, average="macro")
		cc = cohen_kappa_score(self.y_test, y_pred)

		self.myColumns["Algorithm"].append(name)
		self.myColumns["Accuracy"].append(acs)
		self.myColumns["F-Score"].append(fs)
		self.myColumns["Cohen Kappa Score"].append(cc)
		self.myColumns["Precision-Score"].append(ps)
		self.myColumns["Recall-Score"].append(rs)
		self.myColumns["CPU-Training Time"].append(trainingTime)
		self.myColumns["CPU-testing Time"].append(testingTime)

	def runNB(self):
		name = "Naive Bayes Classifier"
		gnb = GaussianNB()
		st = time.clock()
		gnb.fit(self.X_train, self.y_train)
		trainingTime = round(time.clock()-st,6)
		st = time.clock()
		y_pred = gnb.predict(self.X_test)
		testingTime = round(time.clock()-st,6)
		acs = accuracy_score(self.y_test, y_pred)
		fs = f1_score(self.y_test, y_pred, average="macro")
		ps = precision_score(self.y_test, y_pred, average="macro")
		rs = recall_score(self.y_test, y_pred, average="macro")
		cc = cohen_kappa_score(self.y_test, y_pred)

		self.myColumns["Algorithm"].append(name)
		self.myColumns["Accuracy"].append(acs)
		self.myColumns["F-Score"].append(fs)
		self.myColumns["Cohen Kappa Score"].append(cc)
		self.myColumns["Precision-Score"].append(ps)
		self.myColumns["Recall-Score"].append(rs)
		self.myColumns["CPU-Training Time"].append(trainingTime)
		self.myColumns["CPU-testing Time"].append(testingTime)

	def runSVM(self):
		name = "Support Vector Machine"
		svc = LinearSVC()
		st = time.clock()
		svc.fit(self.X_train, self.y_train)
		trainingTime = round(time.clock()-st,6)
		st = time.clock()
		y_pred = svc.predict(self.X_test)
		testingTime = round(time.clock()-st,6)
		acs = accuracy_score(self.y_test, y_pred)
		fs = f1_score(self.y_test, y_pred, average="macro")
		ps = precision_score(self.y_test, y_pred, average="macro")
		rs = recall_score(self.y_test, y_pred, average="macro")
		cc = cohen_kappa_score(self.y_test, y_pred)

		self.myColumns["Algorithm"].append(name)
		self.myColumns["Accuracy"].append(acs)
		self.myColumns["F-Score"].append(fs)
		self.myColumns["Cohen Kappa Score"].append(cc)
		self.myColumns["Precision-Score"].append(ps)
		self.myColumns["Recall-Score"].append(rs)
		self.myColumns["CPU-Training Time"].append(trainingTime)
		self.myColumns["CPU-testing Time"].append(testingTime)

	def runDT(self):
		name = "Decision Trees"
		tree = DecisionTreeClassifier(max_depth = 4)
		st = time.clock()
		tree.fit(self.X_train, self.y_train)
		trainingTime = round(time.clock()-st,6)
		st = time.clock()
		y_pred = tree.predict(self.X_test)
		testingTime = round(time.clock()-st,6)
		acs = accuracy_score(self.y_test, y_pred)
		fs = f1_score(self.y_test, y_pred, average="macro")
		ps = precision_score(self.y_test, y_pred, average="macro")
		rs = recall_score(self.y_test, y_pred, average="macro")
		cc = cohen_kappa_score(self.y_test, y_pred)

		self.myColumns["Algorithm"].append(name)
		self.myColumns["Accuracy"].append(acs)
		self.myColumns["F-Score"].append(fs)
		self.myColumns["Cohen Kappa Score"].append(cc)
		self.myColumns["Precision-Score"].append(ps)
		self.myColumns["Recall-Score"].append(rs)
		self.myColumns["CPU-Training Time"].append(trainingTime)
		self.myColumns["CPU-testing Time"].append(testingTime)

	def runLR(self):
		name = "Logistic Regression"
		lr = LogisticRegression()
		st = time.clock()
		lr.fit(self.X_train, self.y_train)
		trainingTime = round(time.clock()-st,6)
		st = time.clock()
		y_pred = lr.predict(self.X_test)
		testingTime = round(time.clock()-st,6)
		acs = accuracy_score(self.y_test, y_pred)
		fs = f1_score(self.y_test, y_pred, average="macro")
		ps = precision_score(self.y_test, y_pred, average="macro")
		rs = recall_score(self.y_test, y_pred, average="macro")
		cc = cohen_kappa_score(self.y_test, y_pred)

		self.myColumns["Algorithm"].append(name)
		self.myColumns["Accuracy"].append(acs)
		self.myColumns["F-Score"].append(fs)
		self.myColumns["Cohen Kappa Score"].append(cc)
		self.myColumns["Precision-Score"].append(ps)
		self.myColumns["Recall-Score"].append(rs)
		self.myColumns["CPU-Training Time"].append(trainingTime)
		self.myColumns["CPU-testing Time"].append(testingTime)

	def runLDA(self):
		name = "Linear Discriminant Analysis"
		lda = LinearDiscriminantAnalysis()
		st = time.clock()
		lda.fit(self.X_train, self.y_train)
		trainingTime = round(time.clock()-st,6)
		st = time.clock()
		y_pred = lda.predict(self.X_test)
		testingTime = round(time.clock()-st,6)
		acs = accuracy_score(self.y_test, y_pred)
		fs = f1_score(self.y_test, y_pred, average="macro")
		ps = precision_score(self.y_test, y_pred, average="macro")
		rs = recall_score(self.y_test, y_pred, average="macro")
		cc = cohen_kappa_score(self.y_test, y_pred)

		self.myColumns["Algorithm"].append(name)
		self.myColumns["Accuracy"].append(acs)
		self.myColumns["F-Score"].append(fs)
		self.myColumns["Cohen Kappa Score"].append(cc)
		self.myColumns["Precision-Score"].append(ps)
		self.myColumns["Recall-Score"].append(rs)
		self.myColumns["CPU-Training Time"].append(trainingTime)
		self.myColumns["CPU-testing Time"].append(testingTime)


	def runRF(self):
		name = "Random Forest"
		rf = RandomForestClassifier()
		st = time.clock()
		rf.fit(self.X_train, self.y_train)
		trainingTime = round(time.clock()-st,6)
		st = time.clock()
		y_pred = rf.predict(self.X_test)
		testingTime = round(time.clock()-st,6)
		acs = accuracy_score(self.y_test, y_pred)
		fs = f1_score(self.y_test, y_pred, average="macro")
		ps = precision_score(self.y_test, y_pred, average="macro")
		rs = recall_score(self.y_test, y_pred, average="macro")
		cc = cohen_kappa_score(self.y_test, y_pred)

		self.myColumns["Algorithm"].append(name)
		self.myColumns["Accuracy"].append(acs)
		self.myColumns["F-Score"].append(fs)
		self.myColumns["Cohen Kappa Score"].append(cc)
		self.myColumns["Precision-Score"].append(ps)
		self.myColumns["Recall-Score"].append(rs)
		self.myColumns["CPU-Training Time"].append(trainingTime)
		self.myColumns["CPU-testing Time"].append(testingTime)

	def runETC(self):
		name = "Extra Trees"
		etc = ExtraTreesClassifier()
		st = time.clock()
		etc.fit(self.X_train, self.y_train)
		trainingTime = round(time.clock()-st,6)
		st = time.clock()
		y_pred = etc.predict(self.X_test)
		testingTime = round(time.clock()-st,6)
		acs = accuracy_score(self.y_test, y_pred)
		fs = f1_score(self.y_test, y_pred, average="macro")
		ps = precision_score(self.y_test, y_pred, average="macro")
		rs = recall_score(self.y_test, y_pred, average="macro")
		cc = cohen_kappa_score(self.y_test, y_pred)

		self.myColumns["Algorithm"].append(name)
		self.myColumns["Accuracy"].append(acs)
		self.myColumns["F-Score"].append(fs)
		self.myColumns["Cohen Kappa Score"].append(cc)
		self.myColumns["Precision-Score"].append(ps)
		self.myColumns["Recall-Score"].append(rs)
		self.myColumns["CPU-Training Time"].append(trainingTime)
		self.myColumns["CPU-testing Time"].append(testingTime)


	def runAlgorithms(self):
		self.runKNN()
		self.runNB()
		self.runSVM()
		self.runDT()
		self.runLR()
		self.runLDA()
		self.runRF()
		self.runETC()

	def calculateRank(self):
		normperfMatrix = defaultdict(list)
		weights_array = fuzzy_AHP(AHP_features_matrix)
		weights = {}
		for i in range(len(weights_array)):
			weights[features[i]] = weights_array[i]	
		for k in self.myColumns.keys():
			if k!='Algorithm':
				s = sum(i for i in self.myColumns[k])
				normperfMatrix[k] = [j/s for j in self.myColumns[k]]
			else:
				normperfMatrix[k] = self.myColumns[k]
		n = len(normperfMatrix['Algorithm'])
		PIS = []
		NIS = []
		RC = []
		for i in range(n):
			wa = sum(normperfMatrix[k][i]*weights[k] for k in normperfMatrix.keys() if k!='Algorithm')
			RC.append([wa,i])
		normperfMatrix["RC"] = [i[0] for i in RC]
		RC.sort(key = lambda x:x[0],reverse=True)
		normperfMatrix["RANK"] = [0]*n 
		for i in range(1,n+1):
			normperfMatrix['RANK'][RC[i-1][1]] = i

		x = self.y_axis.replace(":","")
		with open("{}_{} algorithms.csv".format(path_leaf(self.dataset).split(".")[0].capitalize(),x), 'w') as f:
			pd.concat([pd.DataFrame(self.myColumns),pd.DataFrame({"":[]}), pd.DataFrame(normperfMatrix)], axis=1).to_csv(f)

		columns["Top Algorithm"].append(self.myColumns["Algorithm"][RC[0][1]])
		pd.DataFrame(columns).to_csv("DatasetsCharacteristics.csv")



def main():
	with open('data.txt') as json_file:
		data = json.load(json_file)
		for x in data.keys():
			print(x)
			d = Dataset("./Datasets/{}".format(x),data[x])
			d.makeCharacteristicsData()
			d.runAlgorithms()
			d.calculateRank()

if __name__=="__main__":
	main()