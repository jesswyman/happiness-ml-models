
import pandas as pd
happiness = pd.read_csv('/content/drive/MyDrive/CS_167_Data/happiness_data.csv')
happiness.head()

#I'm only going to work with my target and predictor variables, and drop everything else, for the sake of keeping my data clean.

subset = happiness[["Social support and Family", "Freedom to make life choices", "Generosity", "Score"]]

#Now I'll check for missing values
subset.isnull().sum()
#There are no missing values in the columns we're using, so no need to drop any observations

#we're using regression, so we don't need to make any dummy variables for our predictors

#now we'll split our data into training and testing sets
target= 'Score'
predictors = subset.columns.drop(target)
train_data, test_data, train_sln, test_sln = train_test_split(subset[predictors], subset[target], test_size = 0.2, random_state=41)
train_data.head()

# build and test a baseline model here

import sklearn
import pandas
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors
from sklearn import dummy

mean_dummy = sklearn.dummy.DummyRegressor(strategy="mean")
mean_dummy.fit(train_data, train_sln)

predicted = mean_dummy.predict(test_data)
actual = test_sln

def mae(actual, predicted):
  mae = sum(abs(actual - predicted))/len(actual)
  return mae

maes = mae(actual, predicted)
print(maes)

#By just predicting based on the mean, we're getting a mean average error of .9237, which is really high (not what we want)


#k-NN - tuning k
k_values = [2, 5, 10, 20, 40, 80, 100, 200, 400, 500]
knn_maes = []

for k in k_values:
  knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=k)
  knn.fit(train_data, train_sln)
  predictions = knn.predict(test_data)
  current_mae = mae(test_sln, predictions)
  knn_maes.append(current_mae)

import matplotlib.pyplot as plt

plt.suptitle('Happiness Data k Tuning',fontsize=18)
plt.xlabel('k')
plt.ylabel('mean absolute error')
plt.plot(k_values,knn_maes,'ro-',label='k-NN')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,500,0,4])

print(knn_maes)

plt.show()
#k=20 seems to be the best option here; it has the lowest mean absolute error (0.6502206928773887), which is what we want.

#weighted k-NN - tuning k
k_values = [2, 5, 10, 20, 40, 80, 100, 200, 400, 500]
knn_maes = []

for k in k_values:
  knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance')
  knn.fit(train_data, train_sln)
  predictions = knn.predict(test_data)
  current_mae = mae(test_sln, predictions)
  knn_maes.append(current_mae)

plt.suptitle('Happiness Data weighted k Tuning',fontsize=18)
plt.xlabel('k')
plt.ylabel('mean absolute error')
plt.plot(k_values,knn_maes,'ro-',label='k-NN')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,500,0,4])

print(knn_maes)

plt.show()
#similarly to the unweighted knn, k=20 seems to be the best here; it has the lowest mean absolute error (0.6471121181329496)
#this is also a slight improvement on the unweighted knn

#decision tree - tuning depth
depth_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
dt_maes = []

for d in depth_values:
  dt = tree.DecisionTreeRegressor(max_depth=d)
  dt.fit(train_data, train_sln)
  predictions = dt.predict(test_data)
  current_mae = metrics.mean_absolute_error(test_sln, predictions)
  dt_maes.append(current_mae)

plt.suptitle('Happiness Data DT Max Depth Tuning',fontsize=18)
plt.xlabel('max depth')
plt.ylabel('mean absolute error')
plt.plot(depth_values,dt_maes,'ro-',label='depth')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,10,0,4])

print(dt_maes)

plt.show()

#best option is max depth = 4; this minimizes our mean absolute error (0.6462170897414596)
#slightly better than both knn models

#random forest - tuning max trees

from sklearn.ensemble import RandomForestRegressor

n_estimator_values = range(1,50)
rf_maes = []

for n in n_estimator_values:

  curr_rf = RandomForestRegressor(n_estimators=n, random_state=41)
  curr_rf.fit(train_data,train_sln)
  curr_predictions = curr_rf.predict(test_data)
  curr_mae = metrics.mean_absolute_error(test_sln,curr_predictions)
  rf_maes.append(curr_mae)


plt.suptitle('Happiness Data DT Max Depth Tuning',fontsize=18)
plt.xlabel('number of trees')
plt.ylabel('mean absolute error')
plt.plot(n_estimator_values,rf_maes,'ro-',label='size')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,49,0,4])

print(rf_maes)

plt.show()

#best option here is max trees = 49; minimizes mae (0.6528346600714222)
#worst model we've seen yet

index = range(len(predictors)) #creates a list of numbers the right size to use as the index

plt.figure(figsize=(8,10)) #making the table a bit bigger so the text is readable
plt.barh(index, rf.feature_importances_,height=0.8) #horizontal bar chart
plt.ylabel('Feature')
plt.yticks(index,predictors) #put the feature names at the y tick marks
plt.xlabel("Random Forest Feature Importance")
plt.show()

#normalized knn

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_data)
train_data_normalized = scaler.transform(train_data)
test_data_normalized = scaler.transform(test_data)

knn = neighbors.KNeighborsRegressor(n_neighbors=20)
knn.fit(train_data_normalized, train_sln)
predictions = knn.predict(test_data_normalized)
print('k-NN')
print('MSE: ', metrics.mean_squared_error(test_sln, predictions))
print('MAE: ', metrics.mean_absolute_error(test_sln, predictions))
print('R2: ', metrics.r2_score(test_sln, predictions))

#normalized weighted knn
knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=20, weights='distance')
knn.fit(train_data_normalized, train_sln)
predictions = knn.predict(test_data_normalized)
print('weighted k-NN')
print('MSE: ', metrics.mean_squared_error(test_sln, predictions))
print('MAE: ', metrics.mean_absolute_error(test_sln, predictions))
print('R2: ', metrics.r2_score(test_sln, predictions))

#normalized decision tree
dt = tree.DecisionTreeRegressor(max_depth=4)
dt.fit(train_data_normalized, train_sln)
predictions = dt.predict(test_data_normalized)
print('decision tree')
print('MSE: ', metrics.mean_squared_error(test_sln, predictions))
print('MAE: ', metrics.mean_absolute_error(test_sln, predictions))
print('R2: ', metrics.r2_score(test_sln, predictions))

#normalized random forest
rf = RandomForestRegressor(n_estimators=49, random_state=41)
rf.fit(train_data_normalized,train_sln)
predictions = rf.predict(test_data_normalized)
print('random forest')
print('MSE: ', metrics.mean_squared_error(test_sln, predictions))
print('MAE: ', metrics.mean_absolute_error(test_sln, predictions))
print('R2: ', metrics.r2_score(test_sln, predictions))



#After training  different machine learning algorithms on this data, I can conclude that the decision tree is the best model for this data set; it has the lowest mean absolute error.
#However, overall the models did not perform that well; all the mean average errors were over .6, which isn't very good in the big picture. Looking at the feature importance chart for the random forest,
#I probably could have made a model with less error using only social support and family, since it seems to be bearing the most significance.

#I tuned the k values for the knns, the max depth for the decision tree, and the number of trees for the random forest. I did this by testing a set of different values to see which values gave me the lowest mean average
#error for the model. I took a set of all the mean average errors and plotted them for each model to better visualize which values minimized the error in my models.

#my tuning graphs are below.

plt.suptitle('Happiness Data k Tuning',fontsize=18)
plt.xlabel('k')
plt.ylabel('mean absolute error')
plt.plot(k_values,knn_maes,'ro-',label='k-NN')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,500,0,4])
plt.show()

plt.suptitle('Happiness Data weighted k Tuning',fontsize=18)
plt.xlabel('k')
plt.ylabel('mean absolute error')
plt.plot(k_values,knn_maes,'ro-',label='k-NN')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,500,0,4])
plt.show()

plt.suptitle('Happiness Data DT Max Depth Tuning',fontsize=18)
plt.xlabel('max depth')
plt.ylabel('mean absolute error')
plt.plot(depth_values,dt_maes,'ro-',label='depth')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,10,0,4])
plt.show()

plt.suptitle('Happiness Data DT Max Depth Tuning',fontsize=18)
plt.xlabel('number of trees')
plt.ylabel('mean absolute error')
plt.plot(n_estimator_values,rf_maes,'ro-',label='size')
plt.legend(loc='lower left', shadow=True)
plt.axis([0,49,0,4])
plt.show()

