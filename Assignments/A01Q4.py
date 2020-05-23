import numpy
import math
total_datasets=1000
dataset=[]
new_dataset=[]
class1_plus_dataset=[]
class1_minus_dataset=[]
labeled_dataset={}      #a dictionary containing two keys corresponding to  the two labels and a list representing each of the labels
w_optimum=[1,0,0,0,0,0,0,0,0,0]
for i in range(total_datasets):
    temp_list = numpy.random.uniform(-1, 1, 10)
    temp_sum = 0
    for i in temp_list:
        temp_sum += i * i
    for i in range(len(temp_list)):
        temp_list[i] /= math.sqrt(temp_sum)
    dataset.append(temp_list)
for i in dataset:
    if abs(numpy.dot(w_optimum,i))<0.1:                                         #eliminating datapoints with gamma < 0.1
        pass
    else:
        new_dataset.append(i)
        if numpy.dot(w_optimum,i)>=0:
            class1_plus_dataset.append(i)
        else:
            class1_minus_dataset.append(i)
#labeled_dataset["Class +1"]=class1_plus_dataset
#labeled_dataset["Class -1"]=class1_minus_dataset

"""
run the following classes to print the labeled datasets which are stored in  dictionary with the appropriate key
"""

print("Class +1 datapoints (",len(class1_plus_dataset),")",labeled_dataset["Class +1"])
print("Class -1 datapoints (",len(class1_minus_dataset),")",labeled_dataset["Class -1"])
