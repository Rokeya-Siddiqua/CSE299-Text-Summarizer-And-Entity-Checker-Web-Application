from flask import Flask
from flask import render_template
from flask import request, url_for
import time

#sumy summary package
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#sumy summary package
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
from spacy.spacy_summarization import text_summarizer

#nltk summary package
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.nltk_summarization import nltk_summarizer

#gensim summary package
#from gensim.summarization import summarize

##################################  text classification start ###########################
#text classification Machine Learning package
import pandas as ps
import numpy as np
import sklearn
import joblib
import os
import matplotlib

#from sklearn.naive_bayes import MultinomialNB

#import six
#import sys
#sys.modules['sklearn.externals.six'] = six
#import mlrose
#sys.modules['sklearn.externals.joblib'] = joblib

#wordcloud

# Load Our CountVectorizer
text_vectorizer = open("models/final_news_cv_vectorizer.pkl","rb")
text_countVectorizer = joblib.load(text_vectorizer)

# Load Our Models
def load_prediction_models(model_file):
	loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_models

# show user the text type from the value calculated by the model
def get_keys(val, myDictionary):
    for key, value in myDictionary.items():
        if val == value:
            return key

##################################  text classification end ###########################

#pdf to text converter package
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io


# Web Scraping Pkg
from bs4 import BeautifulSoup
from urllib.request import urlopen  

app = Flask(__name__)

# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Fetch Text From Url
def get_text(raw_url):
	page = urlopen(url=raw_url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text


# Reading Time
def readingTime(mytext):
    total_words = len([token.text for token in nlp(mytext)])
    #total_words = 5000
    estimatedTime = total_words/200.0
    return estimatedTime

#function for pdf to text converter
def pdf2txt(inPDFfile, outTXTfile):
    inputFile = open(inPDFfile, 'rb')
    resourceManager = PDFResourceManager
    returnData = io.StringIO()
    textConverter = TextConverter(resourceManager, returnData, laparams=LAParams())
    interpreter = PDFPageInterpreter(resourceManager, textConverter)
    #process each page in pdf file
    for page in PDFPage.get_pages(inputFile):
        interpreter.process_page(page)
    
    txt = returnData.getvalue()



@app.route("/")
def home():
    return render_template("homePage.html")


#summary from Text
@app.route("/analyze", methods=['POST', 'GET'])
def analyze():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        final_summary = sumy_summarizer(rawtext)
        summary_reading_time = readingTime(final_summary)
        end = time.time()
        final_time = end-start
    return render_template("homePage.html", ctext = rawtext, final_summary = final_summary, final_time = final_time, final_reading_time = final_reading_time, summary_reading_time = summary_reading_time)



#summary from URL
@app.route("/analyze_url", methods=['POST', 'GET'])
def analyze_url():
    start = time.time()
    if request.method == 'POST':
         raw_url = request.form['raw_url']
         rawtext = get_text(raw_url)
         final_reading_time = readingTime(rawtext)
         final_summary = sumy_summarizer(rawtext)
         summary_reading_time = readingTime(final_summary)
         end = time.time()
         final_time = end-start
         return render_template("homePage.html", ctext = rawtext, final_summary = final_summary, final_time = final_time, final_reading_time = final_reading_time, summary_reading_time = summary_reading_time)


#summary compare
@app.route("/compareSummary")
def compareSummary():
    return render_template("compareSummaries.html")


@app.route('/comparer',methods=['GET','POST'])
def comparer():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        #spacy Summarizer
        final_summary_spacy = text_summarizer(rawtext)
        summary_reading_time = readingTime(final_summary_spacy)
        # Gensim Summarizer
        # #final_summary_gensim = summarize(rawtext)
        final_summary_gensim = sumy_summarizer(rawtext)
        summary_reading_time_gensim = readingTime(final_summary_gensim)
		# NLTK Summarizer
        final_summary_nltk = nltk_summarizer(rawtext)
        summary_reading_time_nltk = readingTime(final_summary_nltk)
		# Sumy Summarizer
        final_summary_sumy = sumy_summarizer(rawtext)
        summary_reading_time_sumy = readingTime(final_summary_sumy)
        
        end = time.time()
        final_time = end-start
        return render_template('compareSummaries.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,final_summary_sumy=final_summary_sumy,summary_reading_time_sumy=summary_reading_time_sumy,summary_reading_time_nltk=summary_reading_time_nltk)

##################################  text classification start ###########################
#Text type
@app.route("/textType")
def textType():
    return render_template("textClassifier.html")


# classify text
@app.route('/classify',methods=['GET','POST'])
def classify():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        final_reading_time = readingTime(rawtext)
        prediction_labels = {'business':0,'tech':1,'sport':2,'health':3,'politics':4,'entertainment':5,'crime':6,'weather forecast':7}
        vect_text = text_countVectorizer.transform([rawtext]).toarray()
    
        # Logistic Regression Classifier
        predictorLR = load_prediction_models("models/newsclassifier_Logit_model.pkl")
        predictionLR = predictorLR.predict(vect_text)
        finalResultLR = get_keys(predictionLR,prediction_labels)
        # st.write(predictionLR)

        # Random Forest Classifier
        predictorRF = load_prediction_models("models/newsclassifier_RFOREST_model.pkl")
        predictionRF = predictorRF.predict(vect_text)
        finalResultRF = get_keys(predictionRF,prediction_labels)
        # st.write(predictionRF)

		# Decision Tree Classifier
        predictorDT = load_prediction_models("models/newsclassifier_CART_model.pkl")
        predictionDT = predictorDT.predict(vect_text)
        finalResultDT = get_keys(predictionDT,prediction_labels)
        # st.write(predictionDT)

		# Naive Bayes Classifier
        #NaiveBayModel = open("models/newsclassifierNBmodel.pkl","wb")
        #clf = MultinomialNB()
        #predictionNB = clf.predict(vect_text)
        #finalResultNB = get_keys(predictionNB,prediction_labels)
        #joblib.dump(clf,NaiveBayModel)

        predictorNB = load_prediction_models("models/newsclassifier_NAIVEBAYES_model.pkl")
        predictionNB = predictorNB.predict(vect_text)
        finalResultNB = get_keys(predictionNB,prediction_labels)
        # st.write(predictionNB)
        
        end = time.time()
        final_time = end-start
        return render_template('textClassifier.html',ctext=rawtext,finalResultLR=finalResultLR,finalResultRF=finalResultRF,finalResultDT=finalResultDT,finalResultNB=finalResultNB,final_time=final_time,final_reading_time=final_reading_time)


##################################  text classification end ###########################

if __name__ == "__main__":
    app.run(debug=True)