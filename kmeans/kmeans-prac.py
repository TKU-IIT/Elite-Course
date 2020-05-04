import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans

X=[]
#elements of X
age = []
income = []
spending_score = []

with open('mall_membership.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')    #csv split by ',' charactor
    for row in csv_reader:
        #print(row)
        X.append([  row[2], row[3], row[4] ])   #get data without id and gender (age, income, spending_score)
        age.append(row[2])
        income.append(row[3])
        spending_score.append(row[4])

X = X[1:]   #Remove legend row
age = age[1:]   #Remove legend row
income = income[1:] #Remove legend row
spending_score = spending_score[1:] #Remove legend row

#type casting
X = np.array(X).astype(int)
age = np.array(age).astype(int)
income = np.array(income).astype(int)
spending_score = np.array(spending_score).astype(int)

ax = plt.axes(projection='3d')  #figure generation as 3d
ax.scatter(age, income, spending_score) #set axises

ax.set_xlabel('Age')    #Legend axis X
ax.set_ylabel('Income') #Legend axis Y
ax.set_zlabel('Spending Score') #Legend axis Z
plt.title("Raw Data")   #Set title
plt.show()  #show the figure
#print(X)
inertia = []    #inertia list
for index in range(2,11):   #use index clusters
    kmeans = KMeans(n_clusters=index)   #set K = 2
    results = kmeans.fit_predict(X) #fitting the predictions
    inertia.append(kmeans.inertia_) #appending the current inertia
    #print(results)

    ax = plt.axes(projection='3d')  #figure generation as 3d

    for x, y, z, r in zip(age, income, spending_score, results):
        ax.scatter(x, y, z, color='C'+str(r))   #scatter result with color Cr(r=1 ... K)
    ax.set_xlabel('Age')    #Legend axis X
    ax.set_ylabel('Income') #Legend axis Y
    ax.set_zlabel('Spending Score') #Legend axis Z
    plt.title("K-Means Stage "+str(index))   #Set title
    plt.show()  #show the figure

# print(inertia)
plt.plot(range(2,11), inertia)  #show inertia and clusters graph
plt.show()
