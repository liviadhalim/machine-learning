import sys
from sklearn.neural_network import MLPClassifier

# get train and test file
trainfile = sys.argv[1]
testfile = sys.argv[2]

# list to put the features and the label
x_train = []
y_train = []
x_test = []
y_test = []

# open train file
with open(trainfile) as f:
    lines = f.readlines()
    for i in range(1, len(lines)):
        splitted = lines[i].strip().split(',')
        y_train.append(float(splitted[0]))
        inputs = []
        for j in range(1,len(splitted)):
            inputs.append(float(splitted[j]))
        x_train.append(inputs)
print (x_train)
mlp = MLPClassifier(hidden_layer_sizes= (50), batch_size = 100,solver='adam',learning_rate_init=0.001 ,max_iter=10)
mlp.fit(x_train, y_train)
print (mlp.score(x_train, y_train))
#open test file
with open(testfile) as f2:
    lines2 = f2.readlines()
    for k in range(1, len(lines2)):
        splitted2 = lines2[k].strip().split(',')
        y_test.append(float(splitted2[0]))
        inputs2 = []
        for m in range(1,len(splitted2)):
            inputs2.append(float(splitted2[m]))
        x_test.append(inputs2)

print (mlp.score(x_test,y_test))
