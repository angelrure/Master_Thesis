import pandas
from scipy.stats import ttest_ind_from_stats
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from itertools import product
import networks
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

### Creates models for VCF data.

def read_data(filter_not_480 = True, n_rows = 180000, use_conservation = False):
	"""Reads the input data"""
	print('Reading the data')
	if use_conservation:
		data = pandas.read_table('/genomics/users/aruiz/GTEX_VCF/vcf_psis/sequence_and_conservations_vcf_multiple_5.0.txt', 
			sep = ',', nrows = n_rows, names = ['Sample','Mutationid','Tissue','Seq','Average'] + 
			[str(x)+'c1' for x in range(1,481)] + [str(x)+'c2' for x in range(1,481)], index_col = 1)
	else:
		data = pandas.read_table('/genomics/users/aruiz/GTEX_VCF/vcf_psis/sequence_and_conservations_vcf_multiple_5.0.txt', 
			sep = ',', nrows = n_rows, names = ['Sample', 'Exon','Mutationid','Tissue','Seq','Average'],
			usecols = ['Sample', 'Exon','Mutationid','Tissue','Seq','Average'], index_col = 1)
	if filter_not_480:
		data = data[data['Seq'].str.len() == 960]
	data['Seq2'] = data.Seq.str[480:]
	data['Seq'] = data.Seq.str[0:480]
	return data

def seq_to_individual(data):
	"""Converts the sequence to a list of features for each position."""
	print('Managing the sequences')
	for i in range(1,481):
		data[str(i)] = data['Seq'].str[i-1]
	data = data.drop('Seq', axis = 1)
	return data

def manage_tissues(data, validate_tissues = True):
	"""Converts the tissue to dummy vectors."""
	print('Managing the tissues')
	data = pandas.get_dummies(data, columns = ['Tissue'])
	if validate_tissues == True:
		tissues = pandas.read_table('general_correct_psis_exons_200_vcf_per_tissue.txt', 
			usecols = ['Tissue'], sep = ',')
		tissues = tissues['Tissue'].unique()
		for tissue in tissues:
			if 'Tissue_'+tissue not in data.columns:
				data['Tissue_'+tissue] = 0
	return data


def filter_and_prepare_data(data, filter_by_average = 0.3):
	"""Prepares the data to be analyzed.
	filter_by_average = minimum ammount of PSI value to be considered"""
	print('Preparating the data')
	if filter_by_average:
		data = data[(data['Average']< filter_by_average) | (data['Average'] > 1 - filter_by_average)]
	data = pandas.concat([sequences_to_kmers(data['Seq'], 5, 3), data], axis = 1)
	data = pandas.concat([sequences_to_kmers(data['Seq2'], 5, 3), data], axis = 1)
	data = manage_tissues(data)
	data.drop(['Seq','Seq2', 'Mutationid', 'Sample'], axis = 1, inplace = True)
	labels = data['Average'].round()
	data.drop('Average', axis = 1, inplace = True)
	return data, labels

def sequences_to_kmers(data, ks, sections = 1):
	"""Converts a series of sequence to kmer counting"""
	kmers = []
	for k in range(1, 1+ks):
		kmers.extend([''.join(x) for x in product('ACTG', repeat = k)])
	kmer_data = pandas.DataFrame(index = data.index)
	if sections == 1:
		for kmer in kmers:
			kmer_data[kmer] = data.str.count(kmer)
		return kmer_data
	if sections == 3:
		kmer_data['Seq_up' + data.name[-1]] = data.str[0:200]
		kmer_data['Seq_ex'+ data.name[-1]] = data.str[200:280]
		kmer_data['Seq_down' + data.name[-1]] = data.str[280:]
		sections = [x for x in kmer_data.columns]
		for section in sections:
			for kmer in kmers:
				kmer_data[kmer + '_'+ section] = kmer_data[section].str.count(kmer)
		kmer_data.drop(sections, inplace = True, axis = 1)
	return kmer_data


def generate_sets(data, labels, avoid_overlap = False,  using_kmers = True, norm = True, train_set_size = 0.8):
	print('Generating sets')
	"""Genereats the sets for the model
	- avoid_overlap: ensure there are no overlaps in the event identifiers.
	- norm: normalized the data.
	- train_set_size = proportion of data to the training set."""
	if using_kmers == False:
		data = pandas.get_dummies(data, columns = [str(x) for x in range(1, 481)])
	data = data.reindex_axis(sorted(data.columns), axis=1)
	if norm:
		columns = data.columns
		index = data.index
		data = normalize(data)
		data = pandas.DataFrame(data, index = index, columns = columns)
	if avoid_overlap == False:
		X_train, X_cvt, y_train, y_cvt = train_test_split(data, labels, train_size = train_set_size, random_state = 0)
		X_CV, X_test, y_CV, y_test = train_test_split(X_cvt, y_cvt, train_size = 0.40, random_state = 0)
		return X_train, y_train, X_CV, y_CV, X_test, y_test
	elif avoid_overlap == True:
		train_samples = round(data.shape[0] * 0.8)
		X_train = data[:train_samples] 
		y_train = labels[:train_samples]
		X_test = data[train_samples:]
		y_test = labels[train_samples:]
		X_CV = X_test
		y_CV = y_test
		return X_train, y_train, X_CV, y_CV, X_test, y_test

def kmer_dense_model(n_rows):
	"""Builds a simple kmer based fully connected model for the prediction of the PSI value"""
	global X_train, y_train, X_CV, y_CV, X_test, y_test, model
	data = read_data(n_rows = n_rows)
	data, labels = filter_and_prepare_data(data)
	X_train, y_train, X_CV, y_CV, X_test, y_test  = generate_sets(data, labels, avoid_overlap = True)
	optimizer = keras.optimizers.adam(lr = 0.0001)
	model = networks.model1_network(data.shape[1], 1)
	model_file = 'saved_models/' + 'delete'
	checkpointer = ModelCheckpoint(filepath= model_file, verbose=0, save_best_only=True)
	history = model.fit(X_train.values, y_train.values, batch_size=230, 
				epochs=100,validation_data=(X_CV.values, y_CV.values), verbose=1,
				callbacks = [checkpointer])
	model.load_weights(model_file)
	model.evaluate(X_test.values,y_test.values)
	model.save(model_file)
	


def lstm_model():
	"""Builds a simple lstm model for the prediction of the psi value"""
	data = pandas.read_table('sequence_and_conservations_vcf_multiple_4.0.txt', 
		sep= ',', nrows = 15000, usecols = [3,4,5], names = ['Seq','Average', 'Std'])
	data = data[data['Std'] < 0.3]
	data.drop('Std', axis = 1, inplace = True)
	data.Seq = data.Seq.str.replace('X', '0')
	data.Seq = data.Seq.str.replace('A', '1')
	data.Seq = data.Seq.str.replace('C', '2')
	data.Seq = data.Seq.str.replace('T', '3')
	data.Seq = data.Seq.str.replace('G', '4')
	data['Seq'] = data.Seq.apply(lambda x: list(x))
	data['Seq'] = data.Seq.apply(lambda x: [float(y) for y in x])
	data['Average'] = data['Average'].round()
	embedding_vecor_length = 960
	top_words = 5
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length))
	model.add(LSTM(50))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
	print(model.summary())
	X_train,y_train = sequence.pad_sequences(data.Seq[0:1500]), data['Average'][0:1500]
	X_cv,y_cv = sequence.pad_sequences(data.Seq[1500:2000]), data['Average'][1500:2000]
	X_test,y_test = sequence.pad_sequences(data.Seq[2000:3000]), data['Average'][2000:3000]
	test_X, test_Y = sequence.pad_sequences(data.Seq[3000:4000]), data['Average'][3000:4000]
	a,b,c,d  = train_test_split(test_X, test_Y)
	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=64)
	print(model.evaluate(a,c))



