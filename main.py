#%%allprogram
import pandas as pd , numpy as np
import matplotlib as plt 
from IPython.display import Image
from sklearn import tree, metrics 
import pydotplus
from sklearn import model_selection
import graphviz

datajarkom = pd.read_csv('dataset/jaringankomputer.csv')
print(datajarkom)

datajarkom['class'], labelname = pd.factorize(datajarkom['class'])
# print(labelname)
#print(datajarkom['class'].unique())

datajarkom['waktu'],_ = pd.factorize(datajarkom['waktu'])
datajarkom['prioritas'],_ = pd.factorize(datajarkom['prioritas'])
datajarkom['paket'],_ = pd.factorize(datajarkom['paket'])
datajarkom['frekuensi'],_ = pd.factorize(datajarkom['frekuensi'])
print(datajarkom)


X = datajarkom.iloc[:,:-1]
y = datajarkom.iloc[:,-1]


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,train_size = 0.8 , random_state = None)

print(X_train)
print(y_train)
print(X_test)
print(y_test)
dtree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = None)
dtree.fit(X_train,y_train)
# print(result)


y_pred = dtree.predict(X_test)
count_misclassified =  (y_test != y_pred).sum()
print('Misclassified samples : {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test,y_pred)
print('accuracy : {:.2f}'.format(accuracy))


nama_fitur= X.columns

dot_data = tree.export_graphviz(dtree, out_file= None, 
                                filled=True, rounded=True,
                                feature_names= nama_fitur,
                                class_names= labelname)

#graph = pydotplus.graph_from_dot_data(dot_data)
graph = graphviz.Source(dot_data)
graph.render("DecisionTree",view= True)
# graph.write_png('tree.png')
# Image(graph.create_png())
