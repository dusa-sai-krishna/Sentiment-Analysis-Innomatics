import numpy as np
import spacy
import re
import inflect
import nltk
from nltk import SnowballStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
import pickle


def text_preprocessing(corpus,flag):
    
    # change  of numbers
    p=inflect.engine()
    corpus=re.sub(r'\d+',lambda x: p.number_to_words(x.group(0)),corpus)
    
    # remove special characters
    corpus=re.sub('[^a-zA-Z]',' ',corpus)
    
    #convert to lower case
    corpus=corpus.lower()
    
    # removal of whitespaces
    corpus=' '.join(corpus.split())

    #tokenize
    words=word_tokenize(corpus)
    if flag=="stemming":
    #stemming
        stemmer=SnowballStemmer(language='english')
        return ' '.join(stemmer.stem(word) for word in words if word not in set(nltk.corpus.stopwords.words('english')))
    else:
    #lemmatization
        lemmatizer=WordNetLemmatizer()
        return ' '.join(lemmatizer.lemmatize(word) for word in words if word not in set(nltk.corpus.stopwords.words('english')))


#flag is either "stemming" or "lemmatization"

def text_vectorizer(text):
    nlp=spacy.load('en_core_web_lg')
    vectorized_text=nlp(text).vector
    vectorized_text=np.stack(vectorized_text)
    with open(r'models/min_max_scaler.pkl', 'rb') as file:
        scaler=pickle.load(file)
        y=scaler.transform([vectorized_text])
    return y


def getSentiment(vector):
    with open(r'models/logistic_regression.pkl', 'rb') as file:
        model = pickle.load(file)
        sentiment=model.predict(vector)
        print(sentiment)
    return sentiment
