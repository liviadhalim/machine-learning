import errno
import glob
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from itertools import islice

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
    
path = '/homes/iws/lhalim/cse446final/flights2train/*.csv'
path2='/homes/iws/lhalim/cse446final/train-y-flights'

path3 = '/homes/iws/lhalim/cse446final/flights2dev/*.csv'
path4='/homes/iws/lhalim/cse446final/train-y-flights'

trainxfiles = sorted(glob.glob(path), key= numericalSort)
testxfiles = sorted(glob.glob(path3), key= numericalSort)

train= []
train_y = []

test = []
test_y = []

stringlabeltrain = []
stringlabeltest = []


for file in trainxfiles: 
    with open(file) as f:
        splitted = f.readline().split(",")
        inputs = []
        for j in range(len(splitted)):
            if j==2:
                stringlabeltrain.append(splitted[j])
            else:
                inputs.append(float(splitted[j]))
        train.append(inputs)

values = np.array(stringlabeltrain)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

for x in range(len(train)):
    train[x].extend(onehot_encoded[x])
    
with open(path2) as f2:
    lines = f2.readlines()
    for l in islice(lines,0,8000):
        train_y.append(float(l.strip()))
        
    
for file in testxfiles: 
    with open(file) as f:
        splitted = f.readline().split(",")
        inputs = []
        for j in range(len(splitted)):
            if j==2:
                stringlabeltest.append(splitted[j])
            else:
                inputs.append(float(splitted[j]))
        test.append(inputs)
        
with open(path4) as f2:
    lines = f2.readlines()
    for l in islice(lines,8000,10044):
        test_y.append(float(l.strip()))

values = np.array(stringlabeltest)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

for x in range(len(test)):
    test[x].extend(onehot_encoded[x])

mlp = MLPRegressor(hidden_layer_sizes= (1000,1000), batch_size = 100,solver='adam',learning_rate_init=0.001 ,max_iter=100)
mlp.fit(train, train_y)
print (mlp.score(train,train_y))
print (mlp.score(test,test_y))

