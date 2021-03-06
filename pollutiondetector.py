import numpy as np
import os, random
import pandas as pd 
from sklearn import cross_validation, ensemble, preprocessing, metrics

class pollution (object):
	def __init__(self, ):
		self.data_path = './data/'
		self.forest = ensemble.RandomForestClassifier(n_estimators = 150)

	def readCsv(self, pathway):
		CsvFile = os.listdir(pathway)
		data = []
		for i in CsvFile:
			try :
				polluSheet = pd.read_csv( self.data_path + i ).drop(['採樣分區','縣市','河川','備註','水體分類等級','測站名稱','採樣日期','測站編號'],axis=1).drop([0]).replace([r'\<',r'\>',r'上午',r'下午'],'',regex=True).replace('--','NaN').replace('NaN','0')
				polluSheet = polluSheet[ polluSheet['河川污染指數']!= '0']
				data.append(polluSheet)
			except:
				continue
		for i in range(1,len(data)):
			data[0] = pd.concat([data[0],data[i]])
		return data[0]

	def DataSplit(self, data):
		titanic_X = data.drop(['河川污染指數'],axis = 1)
		titanic_Y = data['河川污染指數']
		train_X, test_X, train_y, test_y = cross_validation.train_test_split(titanic_X, titanic_Y, test_size = 0.3)
		return train_X, test_X, train_y, test_y

	def TrainRandomForestModel(self, trainX, trainY, testX, testY):
		forest_fit = self.forest.fit(trainX, trainY)
		test_y_predicted = self.forest.predict(testX)
		accuracy = metrics.accuracy_score(testY, test_y_predicted)
		print (accuracy)

	def PredictThePollution(self, data):
		daX = data.drop(['河川污染指數'],axis = 1)
		daY = data['河川污染指數']
		return self.forest.predict(daX), daY

if __name__ == '__main__':
	pollu = pollution()
	train_X, test_X, train_y, test_y = pollu.DataSplit(pollu.readCsv(pollu.data_path))
	pollu.TrainRandomForestModel(train_X, train_y, test_X,test_y)




	
















