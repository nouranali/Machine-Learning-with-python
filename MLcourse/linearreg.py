import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import linear_model

url =('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')
df=pd.read_csv(url)
df.to_csv('FuelConsumptionCo2.csv')
df.head()
df.describe()
cdf= df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('Cyliners')
plt.ylabel('CO2 Emissions')
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test= cdf[~msk]

plt.scatter (train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emissions")
plt.show()

##starting linear regression
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y= np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print('Coeffs:' , regr.coef_)
print ('Intercept: ',regr.intercept_)


##testing the o/p
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")