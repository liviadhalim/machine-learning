import sys
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice

def findLowerMargin(Dataset, marginweight, bias):
    minimum = sys.maxint
    marginList = []
    with open(Dataset) as file:
       lines = file.readlines()
       marginweight = marginweight / np.linalg.norm(marginweight)
       for r in lines:
          marginList = []
          currLine = r.strip().split("\t")
          for k in currLine[1:]:
            marginList.append(k)
          marginList = np.array(marginList).astype('float')
          marginList = np.reshape(marginList, (1, len(marginList)))
          marginweight = np.reshape(marginweight, (len(marginweight),1))
          marginsign = abs(np.dot(np.array(marginList),marginweight))
          if marginsign < minimum:
             minimum = marginsign
    return minimum

# save the input file
trainfile = sys.argv[1]
testfile = sys.argv[2]

# get the number of dimensions of x from first line of file
with open(trainfile) as f:
  firstline = f.readline().split("\t")
  weight = np.zeros(len(firstline)-1)
# make lists for plotting purpose
list_x = []
list_train_error = []
list_test_error = []

with open(trainfile) as myfile:
    head = list(islice(myfile, 10))
    random.shuffle(head)
# training for epochs
for x in range(0,20000):
  # open trainfile
  with open(trainfile) as f:
     lines = f.readlines()
     random.shuffle(lines)
     # append the x for the plot
     list_x.append(x)
     # counter for training data 
     counter = 0
     # counter for test data
     counter2 = 0
     for l in lines:
          # extract the y
          splitted = l.strip().split("\t")
          y = float(splitted[0])
          # extract the x to a matrix
          matrices = []
        	for item in splitted[1:]:
        	    matrices.append(item)
          print matrices
          matrices = np.array(matrices)
          # dot product to check sign
          matrices = matrices.astype('float')
          sign = np.dot(matrices,weight)
          # compare sign
          if (sign <= 0.0 and  y>0.0) or (sign > 0.0 and y <= 0.0) :
              #update weight
              weight = weight + np.array(y) *  matrices
              counter += 1
  # append the error from training for the plot
  list_train_error.append(counter*100.0/len(lines))
  
  # open test file
  with open(testfile) as f2:
     for l2 in f2:
          # extract the y
          splitted2 = l2.strip().split("\t")
          y2 = float(splitted2[0])
          # extract the x to a matrix
          matrices2 = []
          for item in splitted2[1:]:
              matrices2.append(item)
          matrices2 = np.array(matrices2)
          # dot product to check sign
          matrices2 = matrices2.astype('float')
          sign2 = np.dot(matrices2,weight)
          # compare sign
          if (sign2 <= 0.0 and  y2>0.0) or (sign2 > 0.0 and y2 <= 0.0) :
            counter2 += 1
     # append the error from test for the plot 
     list_test_error.append(counter2*100.0/len(lines))

  # print the epoch, number of mistakes made in train file, and number of mistakes made in test file
  print "Epoch: " + str(x)
  print "Training Mistake: " + str(counter)
  print "Test Mistake: " + str(counter2)
#plot the graph
print findLowerMargin(trainfile, weight, 0)   
plt.plot(list_x,list_train_error, color="blue", label="training error rate")
plt.plot(list_x,list_test_error, color="red", label="test error rate")
plt.legend(loc='upper left', frameon=False)
plt.margins(0.05)
plt.xlabel("Epochs")
plt.ylabel("Error rate")
plt.savefig("Perceptron.png")

def main():
    # my code here

  if __name__ == "__main__":
      main()
