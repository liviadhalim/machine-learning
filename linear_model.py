import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import math
from math import e
from itertools import islice

trainfile = sys.argv[1]
testfile = sys.argv[2]
list_x = []
list_x_linear = []
averagedsquarederrorlinear = []


with open(trainfile) as f:
  firstline = f.readline().split(",")
  weight_linear = np.zeros(len(firstline))
  weight_logistical = np.zeros(len(firstline))
  

k =100000
sum_linear = 0.0


for x in range (0,k):
  with open(trainfile) as f:
    head = list(islice(f, 400))
    random.shuffle(head)
    splitted = head[0].strip().split(",")
    label = np.float128(splitted[0])
    matrices = []
    for item in splitted[1:]:
	      matrices.append(item)
    matrices.append(1.0)
    matrices = np.array(matrices)
    matrices = matrices.astype('float128')
            
    with np.errstate(all='raise'):
          try:
                list_x.append(x)
                # linear function
                product_linear = np.dot(weight_linear, matrices)
                lineargradient = -2.0*0.8*(label- (product_linear))*matrices
                weight_linear = weight_linear - lineargradient
                
               
                # development testing using training subset (line 250 - 500)
                with open(trainfile) as f2:
                    dev = list(islice(f2, 400, 500))
                    random.shuffle(dev)
                    splitted2 = dev[0].strip().split(",")
                    label2 = np.float128(splitted2[0])
                    matrices2 = []
                    for item in splitted2[1:]:
                        matrices2.append(item)
                    matrices2.append(1.0)
                    matrices2 = np.array(matrices2)
                    matrices2 = matrices2.astype('float')
                    
                    prediction_linear = np.dot(weight_linear,matrices2)
                    sum_linear += (prediction_linear - label2)**2
                    averagedsquarederrorlinear.append(sum_linear/(x+1))
                    list_x_linear.append(x)
 
          except FloatingPointError:
                print("Overflow Error")

plt.plot(list_x_linear, averagedsquarederrorlinear, color="blue", label="Linear Average Square Error")
plt.legend(loc='upper left', frameon=False)
plt.xlabel("Iterations")
plt.ylabel("Average Square Error")
plt.savefig("Linear.png")

def main():
    # my code here

  if __name__ == "__main__":
      main()
      
