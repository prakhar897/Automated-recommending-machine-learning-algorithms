import pandas as pd
import numpy as np
from collections import *

columns = defaultdict(list)

def make_characteristics_data(dataset):
	df = pd.read_csv(dataset)
	attributes = len(df.count())
	Instance_Count = len(df)
	Numerical_Attributes = len(df.describe().count())
	df['full_count'] = df.apply(lambda x: x.count(), axis=1)
	Missing_Count =(Instance_Count*attributes)-df['full_count'].sum()

	columns["dataset"].append(dataset)
	columns["attributes"].append(attributes)
	columns["Instance_Count"].append(Instance_Count)
	columns["Numerical_Attributes"].append(Numerical_Attributes)
	columns["Missing_Count"].append(Missing_Count)

def make():
	make_characteristics_data("./Datasets/amazon.csv")
	make_characteristics_data("./Datasets/ramen-ratings.csv")
	print(columns)
	df = pd.DataFrame(columns)
	df.to_csv("file.csv")

make()
