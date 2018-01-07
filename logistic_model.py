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
list_x_logistical = []
averagedsquarederrorlogistical = []


with open(trainfile) as f:
  firstline = f.readline().split(",")
  weight_linear = np.zeros(len(firstline))
  weight_logistical = np.zeros(len(firstline))
  

k =100000
sum_linear = 0.0
sum_logistical = 0.0


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
                # logistical function 
                product_logistical = np.dot(weight_logistical, matrices)
                denominator = (1.0+ e**((label)*(product_logistical)))
                logisticalgradient = -(0.00001*label*matrices)/denominator
                logisticalgradient = logisticalgradient.astype(np.float128)
                weight_logistical = weight_logistical  - logisticalgradient
                
               
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

                    prediction_logistical = np.dot(weight_logistical,matrices2)
                    if (prediction_logistical <= 0.0) :
                        prediction_logistical = -1
                    else :
                        prediction_logistical = 1
                    
                    sum_logistical += (prediction_logistical - label2)**2
                    averagedsquarederrorlogistical.append(sum_logistical/(x+1))
                    list_x_logistical.append(x)
                    

 
          except FloatingPointError:
                print("Overflow Error")
                
plt.plot(list_x_logistical, averagedsquarederrorlogistical, color="blue", label="Logistic Average Square Error")
plt.legend(loc='upper left', frameon=False)
plt.xlabel("Iterations")
plt.ylabel("Average Square Error")
plt.savefig("Logistic.png")

def main():
    # my code here

  if __name__ == "__main__":
      main()
      
