import json
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing

#Open file stream mammal.json as read mode
with open('mammal.json', 'r') as f:
    raw_data = json.load(f)

data = []
correct_class = []

for row in raw_data:
    data.append([
            row["gave_birth"],
            row["can_fly"],
            row["live_in_water"],
            row["have_legs"]
            ])
            
    correct_class.append(row["class"])

print("Data:")
print(data)
print(correct_class)
print()

labels = ['yes', 'no', 'sometimes', 'mammals', 'non-mammals']

le = preprocessing.LabelEncoder()
le.fit(labels)

print("Before Encoding:")
print(labels)
print("After Encoding:")
print(list(le.classes_))
print()
#Categorize the data and fitting them into labels
for i in range(len(data)):
    data[i] = le.transform(data[i])
correct_class = le.transform(correct_class)

print("Encoded Data")
print(data)
print(correct_class)
print()

clf = GaussianNB()
clf.fit(data, correct_class)

predict = ['yes', 'no', 'yes', 'no']
predict = le.transform(predict)
result = clf.predict([predict])
result = le.inverse_transform(result)
print("Prediction:")
print(result)
print(clf.predict_proba([predict]))
