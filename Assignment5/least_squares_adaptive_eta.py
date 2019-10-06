import sys
import math
import random


#Read Data File In -----------------------------------------------
datafile = sys.argv[1]
f = open(datafile)
data = []
i = 0
l = f.readline()
while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = f.readline()
    data[i].append(1)
    i += 1
rows = len(data)
cols = len(data[0])
#print( "rows:", rows,  "cols", cols) #---to check data was read in correctly and find dimensions of data
f.close()


#Read Label file in ----------------------------------------------
label_data = sys.argv[2]
f = open(label_data)
labels = {}


l = f.readline()
while(l != ''):
    a = l.split()
    labels[int(a[1])] = int(a[0])
    if(labels[int(a[1])] == 0):
        labels[int(a[1])] = -1
    l = f.readline()
f.close()

#Define dot product function to be used below:-------------------------------
def dot_product(list1, list2):
    dp = 0
    refw = list1
    refx = list2
    for j in range (cols):
        dp += refw[j] * refx[j]
    return dp

#Randomly Initialize w --------------------------------------------------------
w = []
for j in range(0, cols, 1):
    w.append(float(0.02*random.uniform(0, 1) - 0.01))


#dellf descent iteration-------------------------------------------------------
eta = 0.001
stop = 0.001
error = 0

#compute dellf and error
while True:
    dellf = []
    dellf.extend(0 for _ in range(cols))
    prev_iter_error = error
    for i in range(0, rows, 1):
        if (labels.get(i) != None):
            dp = dot_product(w, data[i])
            for j in range(0, cols, 1):
                dellf[j] += float((labels[i]-dp)*data[i][j])

    eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001]
    
    bestobj = 1000000000000

    for k in range(0, len(eta_list), 1):
        eta = eta_list[k]

        #update w----------------------------
        for j in range(0, cols, 1):
            w[j] -= eta * dellf[j]

        #calculate error---------------------
        error = 0
        for i in range(0, rows):
            if (labels.get(i) != None):
                error += max(0, 1 - (labels.get(i)) * dot_product(w, data[i]))

        obj = error
        #evaluate best error vs found error
        if obj < bestobj:
            bestobj = obj
            best_eta = eta

        #update w ------------------------------
        for j in range(0, cols, 1):
            w[j] += eta * dellf[j]
            
    #Defines eta when eta is not define yet (1st iteration) 
    if best_eta != None:
        eta = best_eta

    #update w------------------------------------
    for j in range(0, cols, 1):
        w[j] += eta*dellf[j]
    #print(f'w[j]: {w[j]}')

    #compute error-------------------------------
    error = 0
    for i in range(0, rows, 1):
        if (labels.get(i) != None):
            error += (labels[i] - dot_product(w, data[i]))**2

    if abs(prev_iter_error - error) <= stop:
        break

normw = 0
for j in range(0, cols-1, 1):
    normw += w[j]**2

normw = normw**.5
origin_distance = abs(w[len(w)-1]/normw)
#print('distance from origin: ', origin_distance)

#Prints predictions prediction for unlabeled data
for i in range(0, rows, 1):
    if (labels.get(i) == None):
        dp = dot_product(w, data[i])
        if dp > 0:
            print("1,", i)
        else:
            print("0,", i)
            