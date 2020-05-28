import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import streamlit as st
import numpy as np
import string
from PIL import Image
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score 
import joblib
import pickle as pkl

from helper_code import *

def open_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    return data


def show_statistics(data):
    for language, sentences in data.items():
        
        number_of_sentences = 0
        number_of_words = 0
        number_of_unique_words = 0
        sample_extract = ''
        word_list = ' '.join(sentences).split()
        number_of_sentences = len(sentences)
        number_of_words = len(word_list)
        number_of_unique_words = len(set(word_list))
        sample_extract = ' '.join(sentences[0].split()[0:7])
        
        
        print(f'Language: {language}')
        print('-----------------------')
        print(f'Number of sentences\t:\t {number_of_sentences}')
        print(f'Number of words\t\t:\t {number_of_words}')
        print(f'Number of unique words\t:\t {number_of_unique_words}')
        print(f'Sample extract\t\t:\t {sample_extract}...\n')


def preprocess(text):
    '''
    Removes punctuation and digits from a string, and converts all characters to lowercase. 
    Also clears all \n and hyphens (splits hyphenated words into two words).
    
    '''    
    preprocessed_text = text  
    preprocessed_text = text.lower().replace('-', ' ')
    translation_table = str.maketrans('\n', ' ', string.punctuation+string.digits)
    preprocessed_text = preprocessed_text.translate(translation_table)
    return preprocessed_text


def main():

	st.title("Language Classification using Navie bayes Classifier")    
	st.sidebar.title("Language Classifier")
	st.markdown("Which language is this? ")
	st.sidebar.markdown("Which language is this? ")

	image = Image.open('bayes.jpeg')
	st.image(image, use_column_width=True)

	data_raw = dict()
	data_raw['Slovak'] = open_file('Data/Sentences/train_sentences.sk')
	data_raw['Czech'] = open_file('Data/Sentences/train_sentences.cs')
	data_raw['English'] = open_file('Data/Sentences/train_sentences.en')	

	#show_statistics(data_raw)

	data_preprocessed = {k: [preprocess(sentence) for sentence in v] for k, v in data_raw.items()}

	sentences_train,y_train = [],[]

	for k,v in data_preprocessed.items():
	    for sentence in v:
	        sentences_train.append(sentence)
	        y_train.append(k)

	vectorizer = CountVectorizer()

	x_train=vectorizer.fit_transform(sentences_train)




	data_val = dict()
	data_val['Slovak'] = open_file('Data/Sentences/val_sentences.sk')
	data_val['Czech'] = open_file('Data/Sentences/val_sentences.cs')
	data_val['English'] = open_file('Data/Sentences/val_sentences.en')

	data_val_preprocessed = {k: [preprocess(sentence) for sentence in v ] for k, v in data_val.items()} 

	sentences_val, y_val = [], []

	for k, v in data_val_preprocessed.items():
	    for sentence in v:
	        sentences_val.append(sentence)
	        y_val.append(k)
	x_val= vectorizer.transform(sentences_val)

	class_names= ['Slovak','Czech','English']

	st.sidebar.text("Navie Bayes")
	st.sidebar.subheader("Model Hyperparameters")
	alpha = st.sidebar.number_input("Alpha",0.01,10.0,step=0.01,key="alpha")
	fit = st.sidebar.radio("Fit_prior:",(True,False),key="fit_prior")

	if st.sidebar.button("Train",key="train"):
		st.subheader("Navie bayes")
		model = MultinomialNB(alpha=alpha,fit_prior=fit)
		model.fit(x_train,y_train)
		y_pred = model.predict(x_val)
		accuracy = model.score(x_val,y_val)
		st.write("Accuracy: ",accuracy.round(2))
		st.write("Precision: ",precision_score(y_val,y_pred,labels=class_names,average="weighted").round(2))
		st.write("Recall: ",recall_score(y_val,y_pred,labels=class_names,average="weighted").round(2))
		st.write("f1_score",f1_score(y_val,y_pred,average='weighted').round(3))
		st.subheader("Confusion Matrix")
		plot_confusion_matrix(y_val, y_pred, class_names)	
		st.pyplot()
		joblib.dump(model, 'Data/Models/final_model.joblib')
		joblib.dump(vectorizer, 'Data/Vectorizers/final_model.joblib')
	
	text = st.text_input("Enter your sentence: ")


	if st.button("Predict",key="predict"):
		mod =joblib.load('Data/Models/final_model.joblib')
		vector = joblib.load('Data/Vectorizers/final_model.joblib')
		text = [text]
		text_vectorized = vector.transform(text)
		st.header("Prediction on entered sentence is: "+mod.predict(text_vectorized)[0])





if __name__ == '__main__':
    main()
