import pdb
import numpy as np 
import re
import pandas as pd
import nltk
# nltk.download('all')
import math
import time
import pickle
import glob
import os

from nltk import ngrams
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from nltk.stem import WordNetLemmatizer
from collections import defaultdict



def remove_punc(string):

	punctuations = '''-⎡⎤⎣⎦⎢πε−’–×÷!→ˆ•−–′‘“”\n₹€£()-=+–’[]{};:'"\,<>./?@#$%^&*_~`?.∫'''

	for x in string:
		if x in punctuations:
			string = string.replace(x, " ")

	return re.sub(' +', ' ',string.lower().strip())


def fetch_docs(path):
	list_of_files = glob.glob(os.path.join(path, '*'))
	df = []
	for each_file in list_of_files:
		with open(each_file, 'r') as f:
			txt = f.readlines()
		df.append(" ".join(txt))
	docs = [remove_punc(df[i]) for i in range(len(df))]

	return docs


def load_glove_model(File):
	print("Loading Glove Model")

	glove_model = {}

	with open(File,'r') as f:
		for line in f:
			split_line = line.split()
			word = split_line[0]
			embedding = np.array(split_line[1:], dtype = np.float64)
			glove_model[word] = embedding

	print(f"{len(glove_model)} words loaded!")
	return glove_model


def concept_index(concepts):
	c_to_i = {concept: i for i, concept in enumerate(concepts)}

	return c_to_i


def doc_index(docs):
	d_to_i = {doc: i for i, doc in enumerate(docs)}

	return d_to_i



def build_concept_feature(concepts, gm):
	concept_feature = {}

	for concept in concepts:
		words = concept.split()

		try:
			if len(words) > 1:
				v = [gm[word] for word in words]
				concept_feature[concept] = np.average(v, axis = 0)
			else:
				concept_feature[concept] = gm[concept]
		except:
			concept_feature[concept] = np.zeros(300)

	print("Concept feature extracted!!")
	
	concept_to_index = concept_index(concepts)

	concept_feature_matrix = np.array([concept_feature[i] for i in concepts])
	mat = np.matrix(concept_feature_matrix)
	print("Shape of concept_feature {}".format(mat.shape))
	
	with open('./MOOC-DSA/feature/cf.txt','wb') as f:
		for line in mat:
			np.savetxt(f, line, fmt ='%.6f')

	print("Concept feature file created!")
	return  concept_to_index, concept_feature



def build_doc_feature(docs, concepts, cf):
	doc_feature = {}

	for i, doc in enumerate(docs):
		tmp_list = []
		
		for concept in concepts:
			if concept in doc:
				tmp_list.append(concept)

		if len(tmp_list) != 0:
			v = [cf[item] for item in tmp_list]
			doc_feature[doc] = np.average(v, axis = 0)
		else:
			doc_feature[doc] = np.zeros(300)

	print("Document feature extracted!")
	
	doc_to_index = doc_index(docs)

	doc_feature_matrix = np.array([doc_feature[i] for i in docs])
	mat = np.matrix(doc_feature_matrix)
	print("Shape of document_feature {}".format(mat.shape))

	with open('./MOOC-DSA/feature/df.txt','wb') as f:
		for line in mat:
			np.savetxt(f, line, fmt ='%.6f')

	print("Document feature file created!")
	return doc_to_index, doc_feature
	


def ev_tfidf(docs, concepts, c_i, d_i):

	alpha = 0
	beta = 2.5
	doc_to_concept = np.zeros((len(docs), len(concepts)))
	total_num_docs = len(docs)

	# word count for each document
	word_count = {doc: len(nltk.word_tokenize(doc)) for doc in docs}

	# doc_count--> number of documents where the concept appears
	doc_count = {}
	for concept in concepts:
		count = 0
		for doc in docs:
			if concept in doc:
				count += 1
		doc_count[concept] = count

	#adl-> average document length
	adl = np.mean(list(word_count.values()))
	
	for i, doc in enumerate(docs):
		# print("Processing {} doc".format(i))
		words = nltk.word_tokenize(doc)
		uw = list(np.unique(words))
		atf = len(words) / len(uw)
		dl = len(words)
		
		for concept in concepts:
			# concept_count = doc.count(concept)

			cl = len(concept.split())
			ct = tuple(concept.split())
			doc_ngram = list(ngrams(doc.split(), cl))
			concept_count = doc_ngram.count(ct)

			tf = concept_count / len(words)
			
			x1 = (math.log10(1+tf)) / (math.log10(1 + atf))
			F_x1 = math.exp(-(math.exp(-((x1 - alpha) / beta))))

			x2 = tf * (math.log10(1 + (adl / dl)))
			F_x2 = math.exp(-(math.exp(-((x2 - alpha) / beta))))
			
			IDF = math.log10(total_num_docs / (doc_count[concept] + 1))
			weight = (F_x1 + F_x2) * IDF

			doc_to_concept[d_i[doc],c_i[concept]] = np.around(weight,4)
			
	
	print("Document-concept edge feature extracted!")

	mat = np.matrix(doc_to_concept)
	print("Shape of document-concept-edge_feature {}".format(mat.shape))

	with open('./MOOC-DSA/feature/dcf.txt','wb') as f:
		for line in mat:
			np.savetxt(f, line, fmt ='%.6f')
			
	print("Document-concept edge feature file created!")
			
	return doc_to_concept


def cos_sim(docs, df):

	doc_to_doc = np.zeros((len(docs), len(docs)))

	for i in range(len(doc_to_doc)):
		v1 = df[docs[i]]

		for j in range(len(doc_to_doc[i])):
			v2 = df[docs[j]]
			doc_to_doc[i,j] = np.around(dot(v1, v2) / (norm(v1) * norm(v2) + 1e-10), 6)
	
	print("Document-document edge feature extracted!")

	mat = np.matrix(doc_to_doc)
	print("Shape of document-document-edge_feature {}".format(mat.shape))

	with open('./MOOC-DSA/feature/ddf.txt','wb') as f:
		for line in mat:
			np.savetxt(f, line, fmt ='%.6f')

	print("Document-document edge feature file created!")

	return doc_to_doc


def pmi(docs, concepts):

	window_size = 10
	concept_pairs = []
	count = {}
	concept_to_concept = np.zeros((len(concepts), len(concepts)))
	
	ngs = [list(ngrams(doc.split(), window_size)) for doc in docs]
	windowed_docs = [' '.join(each_gram) for sublist in ngs for each_gram in sublist]
	

	concept_presence_dict = {}
	
	for concept in concepts:

		indices = [i for i, doc in enumerate(docs) if concept in doc]

		concept_presence_dict[concept] = indices

	

	for i, concept_1 in enumerate(concepts):
		for j, concept_2 in enumerate(concepts):

			if i != j :
				concept_1_indices = concept_presence_dict[concept_1]
				concept_2_indices = concept_presence_dict[concept_2]

				p_concept_1 = len(concept_1_indices) / len(windowed_docs) + 1e-10
				p_concept_2 = len(concept_2_indices) / len(windowed_docs) + 1e-10

				p_co_occ = len(list(set(concept_1_indices).intersection(concept_2_indices))) / len(windowed_docs)

				pmi = math.log(p_co_occ / (p_concept_1 * p_concept_2) + 1e-10, 10)
				
				concept_to_concept[i, j] = pmi


	print("Concept-concept edge feature extracted!")
	mat = np.matrix(concept_to_concept)
	print("Shape of concept-concept-edge_feature {}".format(mat.shape))

	with open('./MOOC-DSA/feature/ccf.txt','wb') as f:
		for line in mat:
			np.savetxt(f, line, fmt ='%.6f')

	print("Concept-concept edge feature file created!")
	return concept_to_concept	








def main():

	with open("./MOOC-DSA/data/concepts.txt", 'r') as f:
		all_concepts = f.readlines()
	
	# download glove.6B.300d.txt and put this file path in the glove_path variable
	glove_path = "../../datasets/glove.6B/glove.6B.300d.txt"
	glove_model = load_glove_model(glove_path)

	concepts = [remove_punc(c) for c in all_concepts]
	
	concept_to_index, concept_feature = build_concept_feature(concepts, glove_model)
	
	docs = fetch_docs('./MOOC-DSA/data/docs')
	doc_to_index, doc_feature = build_doc_feature(docs, concepts, concept_feature)
	
	doc_concept_edge_feature = ev_tfidf(docs, concepts, concept_to_index, doc_to_index)

	doc_doc_edge_feature = cos_sim(docs, doc_feature)

	concept_to_concept_edge_feature = pmi(docs, concepts)






if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	total_time = end_time - start_time
	print("total time: {}".format(total_time))

# total time: 55 sec
