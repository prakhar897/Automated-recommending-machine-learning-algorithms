import numpy
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from collections import *
import ntpath
ntpath.basename("a/b/c")

from hyperparameters import AHP_character_matrix,character
from fuzzy_AHP import fuzzy_AHP

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

class SimDataset:

	def __init__(self,dataset,DF):
		self.dataset = dataset
		self.df = pd.read_csv(dataset)
		self.columns = defaultdict(list)
		self.DF = DF

	def getFeatures(self):
		attributes = len(self.df.count())
		instanceCount = len(self.df)
		numericalAttributes = len(self.df.describe().count())
		self.df['full_count'] = self.df.apply(lambda x: x.count(), axis=1)
		missingCount =(instanceCount*attributes)-self.df['full_count'].sum()
		self.df = self.df.drop(columns=["full_count"])

		self.columns["dataset"].append("{}".format(path_leaf(self.dataset).split(".")[0]))
		self.columns["attributes"].append(attributes)
		self.columns["Instance_Count"].append(instanceCount)
		self.columns["Numerical_Attributes"].append(numericalAttributes)
		self.columns["Missing_Count"].append(missingCount)

	def getMaxScore(self):

		weights_array = fuzzy_AHP(AHP_character_matrix)
		weights = {}
		for i in range(len(weights_array)):
			weights[character[i]] = weights_array[i]	
		#print(weights)

		s=0
		for k in self.columns.keys():
			if k != "dataset":
				s += self.columns[k][0]*weights[k]

		a = [(self.columns[k][0]*weights[k])/s for k in self.columns.keys() if k!="dataset"]
		scores = []

		for i,rows in self.DF.iterrows():
			
			my_vec = [rows.attributes,rows.Instance_Count,rows.Numerical_Attributes,rows.Missing_Count]
			#print(my_vec)

			x=0
			for k in range(len(my_vec)):
				x += my_vec[k]*weights_array[k]
			#x = sum(my_vec)
			b = [(my_vec[i]*weights_array[i])/s for i in range(len(my_vec))]

			euc_sim = norm(numpy.array(a)-numpy.array(b))
			euc_sim = 1/(1+euc_sim)
			cos_sim = dot(a, b)/(norm(a)*norm(b))
			avg_sim = round((euc_sim+cos_sim)/2,4)

			scores.append(avg_sim)

		return scores.index(max(scores)), max(scores)

	def getAlgorithm(self):

		idx,score = self.getMaxScore()
		self.columns["Top Algorithm"].append(self.DF["Top Algorithm"][idx])
		if self.columns["dataset"][0] not in self.DF["dataset"]:
			pd.DataFrame(self.columns).to_csv("DatasetsCharacteristics.csv",mode='a', header=False)

		return self.DF["Top Algorithm"][idx]



def main():

	DF = pd.read_csv("./DatasetsCharacteristics.csv")
	with open('datasim.txt') as f:
		data = f.readlines()
		for x in data:
			d = SimDataset("./Datasets/{}".format(x.strip()),DF)
			d.getFeatures()
			ans = d.getAlgorithm()
			print("For the dataset {}, you can use {} for best classification results!".format(x.strip(),ans))


if __name__=="__main__":
	main()


