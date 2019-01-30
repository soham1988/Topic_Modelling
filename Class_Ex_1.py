# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 07:54:52 2019

@author: soham
"""

count=0

loadedjson=open('meta_Clothing_Shoes_and_Jewelry.json','r')

allproducts = {}

listofcategories = {}

for aline in loadedjson:
    count += 1
    if count % 100000 == 0:
        print(count)
    aproduct = eval(aline)
    
    allproducts[aproduct['asin']] = aproduct
    
    for categories in aproduct['categories']:
        for acategory in categories:
            if acategory in listofcategories:
                listofcategories[acategory] += 1
            if acategory not in listofcategories:
                listofcategories[acategory] = 1
                
count=0
allnikeasins = set()

for aproduct in allproducts:
    theproduct = allproducts[aproduct]
    count += 1
    if count % 100000 == 0:
        print (count/1503384)
    for categories in theproduct['categories']:
        for acategory in categories:
            if 'nike' in acategory.lower():
               allnikeasins.add(theproduct['asin'])
                
                
                
loadedjson=open('reviews_Clothing_Shoes_and_Jewelry.json','r')

allreviews = {}
count = 0

for aline in loadedjson:
    count +=1
    if count % 100000 == 0:
        print(count)
    areview = eval(aline)
    theasin = areview['asin']
    thereviewer = areview['reviewerID']
    if theasin in allnikeasins:
        thekey = '%s.%s'%(theasin,thereviewer)
        allreviews[thekey] = areview
        
len(allreviews)

import json
json.dump(allreviews,open('allnikereviews.json','w'))
allreviews=json.load(open('allnikereviews.json','r'))

import nltk
nltk.download('stopwords')
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

stop_words = stopwords.words('english')
stop_words += ['nike']

texts = set()
def load_texts(topicdata):
    for areview in topicdata:
        if 'reviewText' in topicdata[areview]:
            reviewtext = topicdata[areview]['reviewText']
            summary = topicdata[areview]['summary']
            asin = topicdata[areview]['asin']
            review = '%s %s %s'% (asin,summary,reviewtext)
            texts.add(review)
            
print('loading texts')
load_texts(allreviews)

documents = list(texts)

vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(documents) 

true_k = 50

model = KMeans(n_clusters=true_k,max_iter=100000)
model.fit(X)

print('Top terms per cluster')

order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names()

for i in range(true_k):
    topic_terms = [terms[ind] for ind in order_centroids[i,:4]]
    print('%d: %s'%(i,''.join(topic_terms)))
            
import os
outfiles = {}

try:
    os.mkdir('output')

except OSError:
    print('directory already exist')
    
else:
    print("successfully created the directory")
    
for atopic in range (true_k):
    topicterms = [terms[ind] for ind in order_centroids[atopic,:4]]
    outfiles[atopic] = open(os.path.join('output','_'.join(topicterms) + '.txt'),'w')
    
for areview in allreviews:
    if 'reviewText' in allreviews[areview]:
        thereview = allreviews[areview]
        reviewwithmedata = "%s %s %s" % (thereview['asin'], thereview['summary'], thereview['reviewText'])
        Y = vectorizer.transform([reviewwithmedata])
      
        
        for prediction in model.predict(Y):
            outfiles[prediction].write('%s\n'% reviewwithmedata) 
            
for n, f in outfiles.items():
    f.close()




#LDA 
from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import cpu_count

k = 25

lda_tfidf = LatentDirichletAllocation(n_topics = k, n_jobs = cpu_count())
lda_tfidf.fit(X)


#!pip install pyLDAvis
import pyLDAvis
import pyLDAvis.sklearn

pyLDAvis.enable_notebook()


p = pyLDAvis.sklearn.prepare(lda_tfidf, X, vectorizer)
    
pyLDAvis.save_html(p,'pyLDAvis.html')


##################
#               
#outputfile = open('nike.txt', 'w')
#
#outputfile.write(','.join(allnikeasins))
#
#outputfile.close()           

#listofcategories['Vans']  

#
#import spyder_kernels 
#
#spyder_kernels.__version__







#
#with open(meta_Clothing_Shoes_and_Jewelry, encoding='utf-8-sig') as json_file:
#    text = json_file.read()
#    json_data = json.load(json_file)
#    print(json_data)
#    
#listofcategories['nike']   



    