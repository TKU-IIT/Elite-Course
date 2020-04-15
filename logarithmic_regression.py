import csv
with open('graduate_admissions.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        print(row)
