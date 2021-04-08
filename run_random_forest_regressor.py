import pandas
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

dataset = pandas.read_csv("temperature_data.csv")

print(dataset)
print(dataset.shape)
print(dataset.describe())

dataset = pandas.get_dummies(dataset)

print(dataset)
print(dataset.columns)

target = dataset['actual'].values
data = dataset.drop('actual', axis = 1).values

kfold_object = KFold(n_splits = 4)
kfold_object.get_n_splits(data)


test_case = 0
for training_index, test_index in kfold_object.split(data):
	print(test_case)
	test_case = test_case+1
	# print("training: ", training_index)
	# print("test: ", test_index)
	data_training = data[training_index]
	data_test = data[test_index]
	target_training = target[training_index]
	target_test = target[test_index]
	machine = RandomForestRegressor(n_estimators = 201, max_depth = 4)
	machine.fit(data_training, target_training)
	new_target = machine.predict(data_test)
	print(metrics.mean_absolute_error(target_test,new_target))


machine = RandomForestRegressor(n_estimators = 201, max_depth = 4)
machine.fit(data, target)
feature_list = dataset.drop('actual', axis=1).columns
feature_importances_raw = list(machine.feature_importances_)
# print(feature_importances_raw)

feature_importances = [(feature, round(importance,3)) for feature, importance in zip(feature_list, feature_importances_raw)]
# print(feature_importances)
feature_importances = sorted(feature_importances, key = lambda x:x[1], reverse=True)
# print(feature_importances)
[print('{:13} : {}'.format(*i)) for i in feature_importances]


x_values = list(range(len(feature_importances_raw)))
plt.bar(x_values, feature_importances_raw, orientation='vertical')
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('importance')
plt.xlabel('feature')
plt.title('Feature Importance!!')
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()


#######################

from sklearn.tree import export_graphviz
import pydot 

tree = machine.estimators_[4]
export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')













