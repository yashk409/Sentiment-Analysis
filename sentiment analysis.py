
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemma=WordNetLemmatizer()
stopwords=set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews=BeautifulSoup(open('positive.review').read())
positive_reviews=positive_reviews.findAll('review_text')

negative_reviews=BeautifulSoup(open('negative.review').read())
negative_reviews=negative_reviews.findAll('review_text')


def my_tokenizer(s):
    s=s.lower()
    tokens=nltk.tokenize.word_tokenize(s)
    tokens=[t for t in tokens if len(t)>2]
    tokens=[wordnet_lemma.lemmatize(t) for t in tokens]
    tokens=[t for t in tokens if t not in stopwords]
    return tokens

word_index_map={}
current_index=0

positive_tokenized=[]
negative_tokenized=[]


for review in positive_reviews:
    tokens=my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token]=current_index
            current_index+=1

for review in negative_reviews:
    tokens=my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token]=current_index
            current_index+=1


def tokens_to_vector(tokens, label):
    x=np.zeros(len(word_index_map)+1)
    for t in tokens:
        i=word_index_map[t]
        x[i]+=1
    x=x/x.sum()
    x[-1]=label
    return x


N=len(positive_tokenized)+len(negative_tokenized)
data=np.zeros((N,len(word_index_map)+1))


i=0
for tokens in positive_tokenized:
    xy=tokens_to_vector(tokens,1)
    data[i,:]=xy
    i+=1


for tokens in negative_tokenized:
    xy=tokens_to_vector(tokens,0)
    data[i,:]=xy
    i+=1

np.random.shuffle(data)
X=data[:,:-1]
Y=data[:,-1]

xtrain=X[:-500,]
ytrain=Y[:-500,]
xtest=X[-500:,]
ytest=Y[-500:,]

model=LogisticRegression(solver='lbfgs')
model.fit(xtrain,ytrain)
print("Accuracy of Logistic Regression:",model.score(xtest,ytest))

threshold=0.5
for word,index in word_index_map.items():
    weight=model.coef_[0][index]
    if weight>threshold or weight< -threshold:
        print (word,weight)


from sklearn.naive_bayes import MultinomialNB
model2=MultinomialNB()
model2.fit(xtrain,ytrain)
print('Accuracy of MultinomialNB',model2.score(xtest,ytest))


from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(xtrain, ytrain) 

print("Accuracy of SVM",clf.score(xtest,ytest))

