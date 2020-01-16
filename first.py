import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer

train = pd.read_csv('train_E6oV3lV.csv')

test = pd.read_csv('test_tweets_anuFYb8.csv')

#remove (@user)
combi = train.append(test, ignore_index=True, sort=False) #combined data  

def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)

    return input_text

#remove twitter handles(@user)
combi['tidy_tweet']=np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")

#Remove punctuations, Numbers, and Special Characters
combi['tidy_tweet']=combi['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")

#remove short Words
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#print(combi.head())

#tokenize
tokenize_tweet = combi['tidy_tweet'].apply(lambda x:x.split())
#print(tokenize_tweet.head())

stemmer = PorterStemmer()
tokenized_tweet = tokenize_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
#print(tokenized_tweet.head())
#print(len(tokenized_tweet))

#grouping tokens back together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenize_tweet[i])

combi['tidy_tweet'] = tokenized_tweet

#Word Cloud
##all_words = ' '.join([text for text in combi['tidy_tweet']])
##print(all_words[:80])
##from wordcloud import WordCloud
##wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
##plt.figure(figsize=(10,7))
##plt.imshow(wordcloud, interpolation="bilinear")
##plt.axis('off')
##plt.show()

###Words in non racist/sexist tweets
##normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
##
##wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
##plt.figure(figsize=(10, 7))
##plt.imshow(wordcloud, interpolation="bilinear")
##plt.axis('off')
##plt.show()

###Racist/Sexist Tweets
##negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
##wordcloud = WordCloud(width=800, height=500,
##random_state=21, max_font_size=110).generate(negative_words)
##plt.figure(figsize=(10, 7))
##plt.imshow(wordcloud, interpolation="bilinear")
##plt.axis('off')
##plt.show()

# function to collect hashtags
def hashtag_extract(x):
    hashtags =[]
    # loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

#extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label']==0])
#extractinf hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label']==1])
#print(HT_regular)
#unnestig list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative, [])
#print(HT_regular[:10])

###Count graph NON Racist/Sexist Tweets
##a = nltk.FreqDist(HT_regular)
##d = pd.DataFrame({'Hashtag': list(a.keys()),
##                  'Count': list(a.values())})
### selecting top 10 most frequent hashtags     
##d = d.nlargest(columns="Count", n = 10) 
##plt.figure(figsize=(16,5))
##ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
##ax.set(ylabel = 'Count')
##plt.show()
##
###Racist/Sexist Tweets
##b = nltk.FreqDist(HT_negative)
##e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
### selecting top 10 most frequent hashtags
##e = e.nlargest(columns="Count", n = 10)   
##plt.figure(figsize=(16,5))
##ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
##ax.set(ylabel = 'Count')
##plt.show()


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000,stop_words='english')
#bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
#print(bow)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000,stop_words ="english")
#tfdif feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
#print(tfidf)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow=bow[:31962,:]
test_bow = bow[31962:,:]

#splittinf data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42,test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain)

prediction = lreg.predict_proba(xvalid_bow)
#print(prediction)
#print("===")
prediction_int = prediction[:,1]>=0.3
#print(prediction_int)
#print("===")
prediction_int = prediction_int.astype(np.int)
#print(prediction_int)
#pre = lreg.predict(xvalid_bow) gives different answer
#print(pre)
print(f1_score(yvalid, prediction_int))
#print(f1_score(yvalid, pre))


test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False)


train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
lreg.fit(xtrain_tfidf, ytrain)
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int))
