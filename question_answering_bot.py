#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) #downloads packages


# In[3]:


f=open('C:/Users/30697/Desktop/TEST.txt','r',errors = 'ignore')
raw=f.read()
raw = raw.lower() #converts to lowercase to reduce repetition of  words like The and the or When and when


# In[4]:


sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words


# In[5]:


import string
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[6]:


import pandas as pd
from re import search
import sklearn

def response(user_response):
    robot_response=''
    sent_tokens.append(user_response)
    TfidfVec = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = sklearn.metrics.pairwise.cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robot_response=robot_response+"I think I need to read more about that..."
        return robot_response
    else:
        robot_response = robot_response+sent_tokens[idx]
        return robot_response


# In[7]:


import random
GREETING_INPUTS = ("hello", "hi", "whats up","hey")
GREETING_RESPONSES = ["MLCA is the best Course","hello", "hi", "whats up","hey"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[29]:


flag=True
print("MLCA Bot: Hello Mr.Papageorgiou or Mr.Perakis or Me.Fergadis I am MLCA bot. I will try to answer your questions on the guys' project,\nyou can ask me anything. Go ahead please!")
while(flag==True):
  
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye!!'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("MLCA Bot: You're welcome. Be kind with the grades please!")
        else:
            if(greeting(user_response)!=None):
                print("MLCA Bot: "+greeting(user_response))
            else:
                print("MLCA Bot: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("MLCA Bot: take care..")

