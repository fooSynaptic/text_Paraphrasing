from flask import Flask,render_template,request
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from flask import url_for
from nltk.probability import FreqDist
from heapq import nlargest
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.externals import joblib	

from time import time
import json
from jieba import cut
import numpy as np
import re
import copy
import os
import random
import pandas as pd
from collections import Counter

from collections import defaultdict

stopwords = [x.strip() for x in open('/Users/ajmd/Desktop/stopwords.txt')\
      .readlines()]

if not os.path.exists('vector.pkl'):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                max_features=3000,
                                stop_words=stopwords)
else:
    from sklearn.externals import joblib
    vectorizer = joblib.load('vector.pkl')


def factorizer(matrix, n_components, feature_names, n_top_words, tokens, factor = "LDA"):
    t0 = time()
    if factor == "NMF":
        nmf = NMF(n_components=n_components, random_state=1,
            beta_loss='kullback-leibler', 
            solver='mu', max_iter=1000, alpha=.1,
            l1_ratio=.5).fit(matrix)
    elif factor == 'LDA':
        nmf = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0).fit(matrix)
    else:
        return

    nmf_topic = print_top_words(nmf, feature_names, n_top_words)

    #figure out the W and H
    W = nmf.transform(matrix)  #centroids

    H = nmf.components_
    return infer(W, nmf_topic)

def print_top_words(model, feature_names, n_top_words):
    topic_dict = {}

    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        topic_dict[topic_idx] = [feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]]

    #return vocabulary for each topic
    return topic_dict


def infer(res, lda_topics):
    vec = []
    d = {}
    for i, x in enumerate(res):
        topic_tokens = lda_topics[np.argmax(x)]
        tokens = [x for x in vocabs if x in topic_tokens]
        map_rate = round(len(tokens)/len(topic_tokens), 3)
        vec.append(map_rate)
        if np.argmax(x) in d:
            d[np.argmax(x)].append(i)
        else:
            d[np.argmax(x)] = [i]

    res = copy.deepcopy(d)
    for x in res.keys():
        res[x] = [res[x][i] - res[x][i-1] < 3 for i in range(1,len(res[x]))]
        if res[x]:
            res[x] = sum(res[x])/len(res[x])
        else:
            res[x] = 0
    cont_rate = sum([res[x] for x in res.keys() if not x == 0])/(len(res.keys())-1)
    

    re_structured = []
    continue_res = [x[0] for x in Counter(res).most_common(5) if x[0]]
    for idx in continue_res:
        tmp = ''
        begin = 1
        tmp += '\n\n'
        tmp += "Topic" + str(idx)+ '\n\n\n'
        for k in d[idx]:
            if not begin:
                pass
            else:
                tmp += str(k)+'\t'+corpus[k].replace(' ', '') + '\n'
        re_structured.append(tmp)
        
    summary = 'The average performance of this text re-constructure result {0}\
        as: {2} and {1} as: {3}...'.format('topic map rate'.upper(), 'continuity rate'.upper(), \
            round(sum(vec)/len(vec), 4), round(cont_rate,4))
    re_structured.insert(0, summary)
    return sum(vec)/len(vec), cont_rate, d,  re_structured



def summarize(texts,n):
    #get text as a list
    #return a sentence list

    # get feature matrix
    #global sentences
    #sentences = [x.strip() for x in texts.split('\n')]
    #print(type(sentences), len(sentences))
    global corpus
    path= '/Users/ajmd/code/nlp_project/text_multip_class/data/9719_20190128_2300_channel_0.txt'
    corpus = [x['text'] for x in json.loads(open(path).read())]

    sents = [re.sub(r'[0-9 a-z]', '', text) for text in corpus]
    sents = [' '.join(cut(x)) for x in sents]

    global vocabs
    vocabs = set()
    for text in sents: vocabs.update(text.split())

    if os.path.exists('vector.pkl'): 
        feature_M = vectorizer.transform(sents).toarray()
    else:
        feature_M = vectorizer.fit_transform(sents).toarray()
        joblib.dump(vectorizer, 'vector.pkl')

    #decomposition 
    _, _, _, res = factorizer(feature_M, 30, vectorizer.get_feature_names(), 10, vocabs, factor = "NMF")
    return '\n'.join(res)
    



    

app = Flask(__name__)

@app.route("/")
def home():
	return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit():
    text1 = request.form['text']
    text2 = summarize(text1, 1)
    return render_template('home.html', text1=text1, text2=text2)

@app.route('/hiw')
def hiw():
	return render_template('hiw.html')

@app.route('/contact')
def contact():
	return render_template('contact.html')

@app.route("/login")
def login():
	return render_template('login.html')

if __name__ == '__main__':
	app.run(debug=True, port = 5000)