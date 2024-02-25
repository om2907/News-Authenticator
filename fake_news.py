#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import re
import string
import streamlit as st
import pickle


# In[40]:


data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')


# In[41]:


data_true.head()


# In[42]:


data_fake['class'] = 0
data_true['class'] = 1


# In[43]:


data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis = 0, inplace = True)
    
data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis = 0, inplace = True)


# In[44]:


data_fake.shape, data_true.shape


# In[45]:


data_fake_manual_testing['class'] = 0
data_true_manual_testing['class'] = 1


# In[46]:


data_fake_manual_testing.head(10)


# In[47]:


data_true_manual_testing.head(10)


# In[48]:


data_merge = pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)


# In[49]:


data_merge.columns


# In[50]:


data = data_merge.drop(['title', 'subject', 'date'], axis = 1)


# In[51]:


data.isnull().sum()


# In[52]:


data = data.sample(frac = 1)


# In[53]:





# In[55]:


data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)


# In[56]:


data.columns


# In[57]:


data.head()


# In[93]:


def wordopt(text):
    if text is None:
        return ''
    else:
        #text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W", " ", text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>,+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

                  


# In[59]:


data['text'] = data['text'].apply(wordopt)


# In[60]:


x = data['text']
y = data['class']


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)


# In[62]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorization, f)

# In[63]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[64]:


pred_lr = LR.predict(xv_test)


# In[65]:


LR.score(xv_test, y_test)


# In[66]:


print(classification_report(y_test, pred_lr))


# In[67]:


from sklearn .tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[68]:


pred_dt = DT.predict(xv_test)


# In[75]:


DT.score(xv_test,y_test)


# In[76]:


print(classification_report(y_test, pred_dt))


# In[77]:


from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)


# In[79]:


pred_gb = GB.predict(xv_test)


# In[80]:


GB.score(xv_test, y_test)


# In[81]:


print(classification_report(y_test, pred_gb))


# In[83]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)


# In[84]:


pred_rf = RF.predict(xv_test)


# In[85]:


RF.score(xv_test, y_test)


# In[86]:


print(classification_report(y_test, pred_rf))

with open('LR_model.pkl', 'wb') as f:
    pickle.dump(LR, f)

with open('DT_model.pkl', 'wb') as f:
    pickle.dump(DT, f)

with open('GB_model.pkl', 'wb') as f:
    pickle.dump(GB, f)

with open('RF_model.pkl', 'wb') as f:
    pickle.dump(RF, f)

# In[87]:


def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "True news"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF =  RF.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {}  \nGB Prediction: {}  \nRF Prediction: {}". format(output_label(pred_LR[0]),
                                                                                                           output_label(pred_DT[0]),
                                                                                                           output_label(pred_GB[0]),
                                                                                                           output_label(pred_RF[0])))


# In[97]:


news = str(input())
manual_testing(news)


# In[ ]:




    



