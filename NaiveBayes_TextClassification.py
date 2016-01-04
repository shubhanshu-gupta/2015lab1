
# coding: utf-8

# In[2]:

from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier


# In[3]:

newsTrainer = Trainer(tokenizer)


# In[4]:

newsSet =[
    {'text': 'not to eat too much is not enough to lose weight', 'category': 'health'},
    {'text': 'Russia is trying to invade Ukraine', 'category': 'politics'},
    {'text': 'do not neglect exercise', 'category': 'health'},
    {'text': 'Syria is the main issue, Obama says', 'category': 'politics'},
    {'text': 'eat to lose weight', 'category': 'health'},
    {'text': 'you should not eat much', 'category': 'health'}
]


# In[5]:

for news in newsSet:
    newsTrainer.train(news['text'], news['category'])


# In[6]:

newsClassifier = Classifier(newsTrainer.data, tokenizer)


# In[7]:

unknownInstance = "Even if I eat too much, is not it possible to lose some weight"
classification = newsClassifier.classify(unknownInstance)


# In[8]:

print classification


# In[ ]:



