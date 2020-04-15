import requests
import jieba
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import multinomial

filename = "minecraft.txt"

data =[]
r = requests.get("https://www.dcard.tw/_api/forums/minecraft/posts?popular=false&limit=100")
r = r.json()
print("Crawling Minecraft...")
for post in r:
    data.append(['minecraft', post['title']])


with open(filename, 'w+') as outfile:
    for row in data:
        output = row[0]+'\t'+row[1]+'\n'
        outfile.write(output)
