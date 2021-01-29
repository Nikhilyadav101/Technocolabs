

# Natural Language Tool Kit 
from flask import Flask,render_template,session,url_for,redirect
import numpy as np
from wtforms import TextField,SubmitField 
from flask_wtf import FlaskForm 
import re
import nltk

nltk.download('stopwords')

# to remove stopword 
from nltk.corpus import stopwords

# for Stemming propose 
from nltk.stem.porter import PorterStemmer
import joblib



	#loading models
Count_vectorizer = joblib.load("Countvector.sav") #countvectorizer
Classifier = joblib.load("Naive_bayse_model.sav") #NBclassifier



def return_preprocess_data(data):
	corpus = []
	ps = PorterStemmer()

# 17578 (contents) rows to clean

	content = re.sub(r"http\S+", "", data)
	content = re.sub('[^a-zA-Z]', ' ', content)
	content = content.lower()
	
	
	# split to list (delimiter " ") 
	content = content.split()
	
   
	
	# rejoin all string array elements to create back into a string 
	
	content = [ps.stem(word) for word in content if not word in set(stopwords.words('english'))]
	
	# append each string to create 
	# array of clean text 
	content = ' '.join(content)
	corpus.append(content)
	
	return corpus

def return_prediction(Count_vectorizer,Classifier,content):
	#preprocessing the string
	preprocess_data = return_preprocess_data(content)
	
	#using BOW approach to vectorize the data
	vectorised_data = Count_vectorizer.transform(preprocess_data)
	
	#predicting the output using LR classifier
	predict = Classifier.predict(vectorised_data)
	
	if predict[0]==0:
		pred = 'Bot tweet'
	else:
		pred = 'Real Person Tweet'
		
	return pred

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'



class TweetForm(FlaskForm):
	tweet = TextField("Tweet")
	submit = SubmitField("Predict")



@app.route("/",methods = ['GET','POST'])
def index():

	form = 	TweetForm()

	if form.validate_on_submit():

		session['tweet'] = form.tweet.data

		return redirect(url_for("prediction"))
	return render_template('home.html',form=form)	






@app.route('/prediction')

def prediction():

	
	content = session['tweet']
	results = return_prediction(Count_vectorizer,Classifier,content)
	
	return render_template('prediction.html',results = results)   

if __name__ == '__main__':
	
	app.debug = True
	app.run()



   

	
