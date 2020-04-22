import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

raw_data = []
X = []
Y = []

with open('graduate-admissions/graduate_admissions.csv', mode='r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		# print(row)
		raw_data.append(row)

raw_data = raw_data[1:]
# print(raw_data)

for row in raw_data:
	#X[GRE,TOEFL]
	X.append([int(row[1]), int(row[2])])
	#Y->Chance to Admit
	Y.append(int(row[-1]))
# print(X)
# print(Y)

admit_gre = []
admit_toefl = []
reject_gre = []
reject_toefl = []

for (gre, toefl), admit in zip(X,Y):
	if admit == 1:
		admit_gre.append([gre])
		admit_toefl.append([toefl])
	else:
		reject_gre.append([gre])
		reject_toefl.append([toefl])
#two lines
plt.plot(admit_gre, admit_toefl, 'go', label='Admit', alpha=0.8)
plt.plot(reject_gre, reject_toefl, 'ro', label='Reject', alpha=0.8)
#plot Regression data into object
clf = LogisticRegression().fit(X, Y)
print('Coefficients: ', clf.coef_)
print('Intercept: ', clf.intercept_)

#Predict the probability of gre: 320, toefl 100 for admission
print(clf.predict([[320, 100]]))
print(clf.predict_proba([[320, 100]]))

def f(x):
	return -(clf.intercept_/clf.coef_[0][1]) - (clf.coef_[0][0]/clf.coef_[0][1])*x

#Generating the graph
plt.plot([340, 285], [f(340), f(285)])

plt.xlabel('GRE')
plt.ylabel('TOEFL')
plt.legend()
plt.show()
