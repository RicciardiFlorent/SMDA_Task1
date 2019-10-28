# -*- coding: utf-8 -*-

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import nltk
import re
import sklearn 
import sys
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from langdetect import detect
from pycountry import languages
from string import punctuation 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection, naive_bayes, svm
from nltk import NaiveBayesClassifier
from nltk import classify
from sklearn import preprocessing
from random import randint

#nltk.download('stopwords')
#nltk.download('punkt')


emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

tab_lang = []
tab = []

#This function allows to read the data of a csv file and preprocess each tweets
# It returns a Dataframe with all cleaned tweets.     
def get_data(csv_file):
    file = pd.read_csv(csv_file,sep=";")
    Topic = []
    Sentiment = []
    TweetDate = []
    TweetText = []
    cpt = 0
    
    for i in range(len(file)):
        tweet = file['TweetText'][i]
        try:
            
            tweet = tweet.lower()
            tweet = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<ï£¿=>, …’‘?@[\]€^–’_—“`”{|}~'→"""), ' ', tweet) #remove punctuation
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL',  tweet) # remove URLs
            tweet = re.sub('@[^\s]+', 'AT_USER',  tweet) # remove usernames
            tweet = re.sub(r'#([^\s]+)', r'\1',  tweet) # remove the # in #hashtag
            tweet = re.sub('rt|cc|RT|CC', '', tweet)#remove rt and cc
            tweet = re.sub(" \d+", ' ', tweet)#remove digits
            tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
            tweet = emoji_pattern.sub(r'', tweet)#remove emojis
            tweet = re.sub(' \s', ' ', tweet)#remove extra whitespace

            language  = detect(tweet)# detect the language of the tweets
            language_name = languages.get(alpha_2= language).name#convert language in good format
            try:
                stop_words = set(stopwords.words(language_name) + ['AT_USER', 'URL']) 
                tweet = word_tokenize(tweet)
                stemmer = PorterStemmer()
                tweet = [word for word in tweet if word.isalpha()]#remove non alpha carac
                tweet = [stemmer.stem(word) for word in tweet]#stemming of the tweet
                tweet = [word for word in tweet if word not in stop_words]#remove the stopwords
                tweet = ' '.join(tweet)
                if(tweet != ''):
                    Topic.append(file['Topic'][i])
                    Sentiment.append(file['Sentiment'][i])
                    TweetDate.append(file['TweetDate'][i])
                    TweetText.append(tweet)
            except:
                tab.append(tab_lang)

            
        except:
            cpt  += 1            
               
    data = {'Topic' : Topic, 'Sentiment': Sentiment, 'TweetDate': TweetDate, 'TweetText': TweetText}
    
    return pd.DataFrame(data)

def RFC_classifier(trainX, trainY, testX, testY):
    classifier = RandomForestClassifier(n_estimators=100, random_state = 0, max_depth=100)
    classifier.fit(trainX, trainY)
    classifier.fit(testX, testY)
    predictions_RFC = classifier.predict(testX)
    return predictions_RFC


def NB_classifier(trainX, trainY, testX, testY):
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(trainX,trainY)
    Naive.fit(testX, testY)
    predictions_NB = Naive.predict(testX)
    return predictions_NB


def SVM_classifier(trainX, trainY, testX, testY):
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(trainX,trainY)
    SVM.fit(testX, testY)
    predictions_SVM = SVM.predict(testX)
    return predictions_SVM

#this function show the topic and sentiment prediction for a given tweet
def topic_prediction_tweet(predictions_topic,predictions_senti,data):
    nb = randint(0,len(predictions_topic))
    print("\nTopic prediction")
    print("Tweet: "+ data['TweetText'][nb])
    print("Topic: "+ data['Topic'][nb])
    print("Topic prediction: " + predictions_topic[nb])
    print("Sentiment: "+ data['Sentiment'][nb])
    print("Sentiment prediction: " + predictions_senti[nb])

#this function shows for a given topic the sentiment prediction
def senti_prediction_for_topic(topic,data, predictions_senti):
    if(len(data) == len(predictions_senti)):
        count = 0
        count_pos = 0
        count_neg =0
        count_irre = 0
        count_neutral = 0
        for i in range(len(data)):
            if((data['Topic'][i]) == topic):
                count += 1;
                if(data['Sentiment'][i] == 'positive'):
                    count_pos += 1
                elif ((data['Sentiment'][i] == 'negative')):
                    count_neg += 1
                elif ((data['Sentiment'][i] == 'neutral')):
                    count_neutral += 1   
                else:
                    count_irre += 1
        
        res_pos = (count_pos/count)*100
        res_neg = (count_neg/count)*100      
        res_neutral = (count_neutral/count)*100      
        res_irre = (count_irre/count)*100 
        print("\nTweets about %s" % topic)
        print("Positive: %f %% " % res_pos)
        print("Negative: %f %%  " % res_neg)
        print("Neutral: %f %%" % res_neutral)
        print("Irrelevant: %f %% " % res_irre)


                
train = get_data("Train.csv")
test = get_data("Test.csv")
lst_topic = set(train['Topic'])
Train_Y_senti = train['Sentiment']
Test_Y_senti = test['Sentiment']

Train_Y_topic = train['Topic']
Test_Y_topic = test['Topic']

Tfidf_vect = TfidfVectorizer(max_features=500, min_df=5, max_df=0.7)
Train_X_Tfidf = Tfidf_vect.fit_transform(train['TweetText']).toarray()
Test_X_Tfidf = Tfidf_vect.fit_transform(test['TweetText']).toarray()

RFC_pred_senti = RFC_classifier(Train_X_Tfidf, Train_Y_senti, Test_X_Tfidf, Test_Y_senti)
NB_pred_senti = NB_classifier(Train_X_Tfidf, Train_Y_senti, Test_X_Tfidf, Test_Y_senti)
SVM_pred_senti = SVM_classifier(Train_X_Tfidf, Train_Y_senti, Test_X_Tfidf, Test_Y_senti)

RFC_pred_topic = RFC_classifier(Train_X_Tfidf, Train_Y_topic, Test_X_Tfidf, Test_Y_topic)
NB_pred_topic = NB_classifier(Train_X_Tfidf, Train_Y_topic, Test_X_Tfidf, Test_Y_topic)
SVM_pred_topic = SVM_classifier(Train_X_Tfidf, Train_Y_topic, Test_X_Tfidf, Test_Y_topic)

while(1):
    print("\n---------Sentiment analysis program----------")
    print("1 - Accuracy scores and F1_scores")
    print("2 - Topic prediction for a given tweet")
    print("3 - Scores for a given topic (Apple, Microsoft, Google, Twitter")
    print("4 - Exit")
    
    choice = input("Enter your choice : ")
    print(choice)
    if(choice == '1'):
        print('\nSentiment predictions :')
        print("RandomForestClassifier Accuracy Score -> ",accuracy_score(RFC_pred_senti, Test_Y_senti)*100)
        print("RandomForestClassifier F1_Score -> ",f1_score(RFC_pred_senti, Test_Y_senti, average='macro')*100)
        print("Naive Bayes Accuracy Score -> ",accuracy_score(NB_pred_senti, Test_Y_senti)*100)
        print("Naive Bayes F1_Score -> ",f1_score(NB_pred_senti, Test_Y_senti, average='macro')*100)
        print("SVM Accuracy Score -> ",accuracy_score(SVM_pred_senti, Test_Y_senti)*100)
        print("SVM F1_Score -> ",f1_score(SVM_pred_senti, Test_Y_senti, average='macro')*100)

        print('\nTopic predictions :')
        print("RandomForestClassifier Accuracy Score -> ",accuracy_score(RFC_pred_topic, Test_Y_topic)*100)
        print("RandomForestClassifier F1_Score -> ",f1_score(RFC_pred_topic, Test_Y_topic, average='macro')*100)
        print("Naive Bayes Accuracy Score -> ",accuracy_score(NB_pred_topic, Test_Y_topic)*100)
        print("Naive Bayes F1_Score -> ",f1_score(NB_pred_topic, Test_Y_topic, average='macro')*100)
        print("SVM Accuracy Score -> ",accuracy_score(SVM_pred_topic, Test_Y_topic)*100)
        print("SVM F1_Score -> ",f1_score(SVM_pred_topic, Test_Y_topic, average='macro')*100)

    elif choice == '2':
        topic_prediction_tweet(RFC_pred_topic,RFC_pred_senti,test)
    elif choice == '3':
        topic =  input("Enter the topic : ")
        if topic not in lst_topic:
            print('error')
        elif topic in lst_topic:
            senti_prediction_for_topic(topic,test, RFC_pred_senti)
    elif choice == '4':
        sys.exit()
    else:
        print("Error...")
