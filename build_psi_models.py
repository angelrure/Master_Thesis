import pandas
import build_vcf_models as bm
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint  
import networks
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import keras
import random

"""This code is used to perform several actions and analysis with PSI or gene expression data"""

## - Reading and preparaing

def read_psis(n_cols = 7000, nrows = 1000, transpose = True, drop_na = True, source = 'GTEX'):
	print('Reading PSI data')
	"""Reads skipping psi values.
	-n_cols = number of columns to read.
	-n_rows = number of rows to read.
	-transpose: True if the features are not in the columns
	-drop_na: clean the data from missing values.
	-source: the sources of the data (GTEX or TCGA)
	"""
	if source == 'GTEX':
		print('Reading from GTEX')
		data = pandas.read_table('correct_psis.txt', usecols = range(0, min(7862, n_cols)), nrows =  min(37706,nrows))
		data = data.loc[~data.index.duplicated(keep= 'first')]
	if drop_na:
		data = data.dropna(1)
	if transpose:
		data = data.transpose()
	return data

def read_all_psis(n_cols = 7000, nrows = 1000, transpose = True, drop_na = True, source = 'GTEX', skiprows = False):
	print('Reading ALL PSI data')
	"""Reads all psi values.
	-n_cols = number of columns to read.
	-n_rows = number of rows to read.
	-transpose: True if the features are not in the columns
	-drop_na: clean the data from missing values.
	-source: the sources of the data (GTEX or TCGA)	"""
	if source == 'GTEX':
		print('Reading from GTEX')
		if skiprows:
			data = pandas.read_table('/projects_rg/GTEX/data/gtex_events_v7.psi', usecols = [0] + [x for x in range(7863) if x not in skiprows or x == 0], nrows =  min(163596,nrows))
		else:
			data = pandas.read_table('/projects_rg/GTEX/data/gtex_events_v7.psi', usecols = range(0, min(7862, n_cols)), nrows =  min(163596,nrows))
		data = data.loc[~data.index.duplicated(keep= 'first')]
		data = data.fillna(data.mean())
	if drop_na:
		data = data.dropna(1)
	if transpose:
		data = data.transpose()
	return data

def read_TPM(n_cols = 10e10, nrows = 1000, transpose = False, drop_na = True, source = 'TCGA'): # data in ['GTEX','TCGA']
	"""Reads tpm values.
	-n_cols = number of columns to read.
	-n_rows = number of rows to read.
	-transpose: True if the features are not in the columns
	-drop_na: clean the data from missing values.
	-source: the sources of the data (GTEX or TCGA)	"""
	print('Reading TPM data of isoforms')
	if source == 'GTEX':
		data = pandas.read_table('/genomics/users/aruiz/psis/GTPX_TPM_transpose.txt', 
			usecols = range(0, n_cols), nrows = nrows, index_col = 0)
	elif source == 'TCGA':
		data = pandas.read_table('/genomics/users/aruiz/TCGA/TcgaTargetGtex_rsem_isoform_tpm', 
			usecols = range(0, nrows), nrows = n_cols, index_col = 0)
		data = 2**data - 0.001
		data = data.transpose()
	data = data.reindex_axis(sorted(data.columns), axis=1)
	if drop_na:
		data = data.dropna(1)	
	if transpose:
		data = data.transpose()
	return data

def read_gene_TPM(n_cols=100, nrows = 10e10,source = 'GTEX', skiprows = False):
	"""Reads tpm values.
	-n_cols = number of columns to read.
	-n_rows = number of rows to read.
	-transpose: True if the features are not in the columns
	-drop_na: clean the data from missing values.
	-source: the sources of the data (GTEX or TCGA)	"""	
	if skiprows:
		data = pandas.read_table('/genomics/users/aruiz/psis/gtex_RSEM_gene_tpm', nrows = min(n_cols, 60498), 
			usecols = [0] + [x for x in range(7863) if x not in skiprows or x == 0], index_col = 0)		
	else:
		data = pandas.read_table('/genomics/users/aruiz/psis/gtex_RSEM_gene_tpm', nrows = min(n_cols, 60498), 
			usecols = list(range(0, min(int(nrows),7863))), index_col = 0)
	data = data.transpose()
	data = 2**data - 0.00099
	data = data.round(3)
	data = pandas.DataFrame(bm.normalize(data), index = data.index, columns = data.columns)
	return data

def recover_sequence_from_exons(exons):
	"""Reads and recovers the sequence from the skipping exons"""
	exons = pandas.read_table('exons_sequence_200_40_40_200.txt', names = ['Exon_ID', 'Exon_Seq', 'Length'], index_col = 'Exon_ID')

def recover_tissue_information(data, substitute = True, source = 'GTEX'): # source in ['GTEX', 'TCGA']
	"""Merges the tissue information wit the patient information
	-substitute: replaces '-' with '.' to keep consistency.
	-source: the sources of the data (GTEX or TCGA)"""
	print('Recovering tissue information')
	if source == 'GTEX':
		pheno = pandas.read_table('GTEX_phenotype', index_col = 0)
		if substitute:
			pheno.index = pheno.index.str.replace('-', '.')
		data = data.join(pheno['_primary_site'], how = 'inner')
	elif source == 'TCGA':
		pheno = pandas.read_table('/genomics/users/aruiz/TCGA/TcgaTargetGTEX_phenotype.txt', encoding = 'latin-1', index_col = 0)
		data = data.join(pheno[['_primary_site','_sample_type']])
	return data

def filter_by_minimum_number_of_samples_per_tissue(data,minim = 50):
	"""Flters the data by having a minimum amount of samples in each tissue
	data: data to filter.
	minim: minimum ammount of tissues to avoid discarding them."""
	tissues = data.groupby('_primary_site')['_primary_site'].count()
	tissues = tissues[tissues > minim]
	data[data['_primary_site'].isin(tissues)]
	data = data[data['_primary_site'].isin(tissues.index)]
	return data

def read_psi_and_recover_tissue(n_samples, n_variables = 10000, data_type = 'psi', 
filter_tissues = False,ensure_all_tissues = True, source = 'GTEX', label_data = 'Tissue', skiprows = []): #data_type in ['psi', 'tpm']; tpm_data in ['GTEX','TCGA'] label_data in ['Tumor', 'Tissue']
	"""Reads the data (can be diffeerent from psi) and recovers the tissue information. The flags are
	quite self explanatory."""
	if data_type == 'psi':
		data = read_psis(nrows = n_variables, n_cols = n_samples, source = source)
	elif data_type == 'allpsi':
		data = read_all_psis(nrows = n_variables, n_cols = n_samples, source = source, skiprows = skiprows)
	elif data_type == 'tpm':
		data = read_TPM(nrows = n_samples, n_cols = n_variables, source = source)
	elif data_type == 'gene':
		data = read_gene_TPM(nrows = n_samples, n_cols = n_variables, source = source, skiprows =skiprows)
	if (source == 'GTEX') and (data_type == 'psi'):
		data = recover_tissue_information(data, substitute = True, source = source)
	else:
		data = recover_tissue_information(data, substitute = False, source = source)
	if filter_tissues:
		data = filter_by_minimum_number_of_samples_per_tissue(data)
	if label_data == 'Tumor':
		data = data[(data._sample_type == 'Primary Tumor') | (data._sample_type == 'Normal Tissue')]
		labels = data['_sample_type']
		data.drop(['_primary_site', '_sample_type'], axis = 1, inplace = True, errors = 'ignore')
	elif label_data == 'Tissue':
		labels = data['_primary_site']
		data.drop(['_primary_site', '_sample_type'], axis = 1, inplace = True, errors = 'ignore')
	labels = pandas.get_dummies(labels)
	if ensure_all_tissues:
		if source == 'GTEX':
			print('Retrieved base')
			base_tissues = pandas.read_table('GTEX_phenotype', index_col = 0, usecols = ['Sample','_primary_site'])
		elif source == 'TCGA':
			base_tissues = pandas.read_table('/genomics/users/aruiz/TCGA/TcgaTargetGTEX_phenotype.txt', encoding = 'latin-1', index_col = 0)
		for tissue in base_tissues.dropna()._primary_site.unique():
			if tissue not in labels.columns and tissue != ' nan':
				labels[tissue] = 0.0
	labels = labels.reindex_axis(sorted(labels.columns), axis=1)
	#data = data.reindex_axis(sorted(data.columns), axis=1)
	return data, labels

def generate_sets(data, labels, norm = False, do_not_split = False):
	"""Generate sets for the training, validating and testing
	norm for normalize the data.
	do_not_split if you want all the data in the same set, but shuffled. 
	"""
	print('generating sets')
	if norm:
		data = pandas.DataFrame(bm.normalize(data), index = data.index, columns = data.columns)
	print(data.shape, labels.shape)
	if do_not_split:
		data = bm.shuffle(data)
		labels = labels.loc[data.index]
		print('sets generated')
		return data, labels
	X_train, X_cvt, y_train, y_cvt = bm.train_test_split(data, labels, train_size = 0.75, random_state = 0)
	X_CV, X_test, y_CV, y_test = bm.train_test_split(X_cvt, y_cvt, train_size = 0.50, random_state = 0)
	print('sets generated')
	return X_train, y_train, X_CV, y_CV, X_test,y_test

def build_pca(n_samples = 10e10, n_variables = 10000, data_type = 'psi', filter_tissues = True,
	ensure_all_tissues = True):
	"""Builds a simple pca"""
	data, labels = read_psi_and_recover_tissue(n_samples = n_samples, n_variables = n_variables, data_type = data_type, filter_tissues = filter_tissues,ensure_all_tissues = ensure_all_tissues)
	print('Computing PCA')
	pca = PCA()
	pca.fit(data)
	return pca

def get_PCA_data(pca, labels, data):
	"""Performs a simple PCA transformation with the data and a pre-computed pca"""
	projected = pca.transform(data)
	projected = bm.pandas.DataFrame(projected, columns = ['PC'+ str(x) for x in list(range(1, projected.shape[1]+1))])
	projected['labels'] = labels.values
	pca_data = projected.groupby('labels')
	return pca_data


def extract_importance_from_pca(pca, data, PC):
	"""Retrieves the feature importance from a pca"""
	weights = pca.components_[PC-1]
	weights = bm.pandas.DataFrame(weights, data.columns, columns = ['weight'])
	weights.plot(linewidth = 0.1)
	bm.plt.xlabel('Exons')
	bm.plt.ylabel('Absolute importance')
	bm.plt.title('Importance per event for PC' +str(PC))
	bm.plt.show()

def extract_importance_from_single_layer_model_by_tissue(data, labels, model): 
	""" Takes a single layer NN model and extracts the feature importance"""
	weights = model.get_weights()[0]
	weights = pandas.DataFrame(weights, columns = labels.columns, index = data.columns)
	return weights

def extract_feature_importance_single_layer_NN(weights):
	""" Given a dataframe of weights of a single layer NN it extracts the events that are more and least expressed for each tissue"""
	maxes = pandas.DataFrame(index = range(0,50))
	mines = pandas.DataFrame(index = range(0,50))
	for tissue in weights.columns:
		maxes[tissue] = weights[tissue].sort_values(ascending = False)[:50].index
		mines[tissue] = weights[tissue].sort_values()[:50].index
	maxes.drop('<not provided>', axis = 1, inplace = True)
	mines.drop('<not provided>', axis = 1, inplace = True)
	return maxes, mines

def extract_unique_characterizing_exons(set_of_events, tissue):
	"""Given a set of maximum/minimum events and a tissue, it return the events which are unique to the tissue."""
	set_of_events_without_target_tissue = set_of_events.drop(tissue, axis = 1)
	unique_in_target_tissue = [exon for exon in set_of_events[tissue] if exon not in set_of_events_without_target_tissue.values]
	return unique_in_target_tissue

def sequences_to_kmers(data, ks, sections = 1):
	"""Converts a series of sequence to kmer counting
	- ks means the maximum lenght of the k-mer counting.
	- sections can be 1 or 3. Depending on in how many sections we want to devide the sequence. 	
	"""
	kmers = []
	for k in range(1, 1+ks):
		kmers.extend([''.join(x) for x in product('ACTG', repeat = k)])
	kmer_data = pandas.DataFrame(index = data.index)
	if sections == 1:
		for kmer in kmers:
			kmer_data[kmer] = data.str.count(kmer)
		return kmer_data
	if sections == 3:
		kmer_data['Seq_up'] = data.str[0:200]
		kmer_data['Seq_ex'] = data.str[200:280]
		kmer_data['Seq_down'] = data.str[280:]
		sections = [x for x in kmer_data.columns]
		for section in sections:
			for kmer in kmers:
				kmer_data[kmer + '_'+ section] = kmer_data[section].str.count(kmer)
		kmer_data.drop(sections, inplace = True, axis = 1)
	return kmer_data


## -- Analyze

def build_predicting_tissue_model(X_train, y_train, X_CV, y_CV, X_test,y_test,
	model_name = 'tissue_classifier_from_correct_tpm', model_shape = 'simple', batch_size = 512,
	return_model = False, epochs = 2000):
	"""Performs a deep learning model to predict the tissue of the samples. It uses the data and labels seaparated for the traning, 
	cross validation and test sets. 
	- model_name = name you want your model file to have.
	- model_shape = neural network shape. See the network file for more details.
	- batch size: the batch size for the trainign of the model.
	- epochs: how many epochs to train the model."""
	print('Building model')
	model_file = 'saved_models/' + model_name + model_shape
	if model_shape == 'simple':
		model = bm.create_simple_NN(X_CV.shape[1], y_CV.shape[1], activation = 'softmax')
	if model_shape == 'single_layer':
		model = bm.single_layer_NN(X_CV.shape[1], y_CV.shape[1], activation = 'softmax')
	if model_shape == 'custom':
		model = networks.plain_NN(X_CV.shape[1], y_CV.shape[1], 3, 300, halve_each_layer = True)
		print(model.summary())
	if model_shape == 'model1':
		model = networks.model1_network(X_CV.shape[1], y_CV.shape[1])
	if return_model == True:
		return model
	checkpointer = bm.ModelCheckpoint(filepath=model_file,  verbose=0, save_best_only=True)
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_CV, y_CV), verbose=1, callbacks = [checkpointer])
	model.load_weights(model_file)
	model.save(model_file)
	model.evaluate(X_test, y_test)
	return history
	

def plot_by_group(data,first_dimension, second_dimension, kind_of_summary = 'tSNE'): # kind_of_summary in ['PCA', 'MDS', 'tSNE']
	"""
	Plots the two dimensions of choose from a multidimensional data reduction/representation.
	first_dimension: integer. The first dimension to plot from the summary model.
	second_dimension: integer. The second dimension to plot from the summary model.
	kind_of_summary: whether to perform a PCA or a tSNE.
	"""
	if kind_of_summary == 'PCA': 
		first_dimension = 'PC' + str(first_dimension)
		second_dimension = 'PC' + str(second_dimension)
	for label, group in data:
		bm.plt.scatter(group[first_dimension], group[second_dimension], label = label)
	bm.plt.legend(prop = {'size' : 15})
	bm.plt.xlabel(first_dimension, fontsize = 20)
	bm.plt.ylabel(second_dimension, fontsize = 20)
	bm.plt.tick_params(labelsize = 20)
	bm.plt.title(kind_of_summary + ' analysis for all the PSI events by tissue', fontsize = 20)
	bm.plt.show()

def plot_by_group_3d(data,first_dimension, second_dimension, third_dimension, kind_of_summary = 'MDS'): # kind_of_summary in ['PCA', 'MDS']
	"""
	Plots the three dimensions of choose from a multidimensional data reduction/representation.
	"""
	fig = bm.plt.figure()
	ax = Axes3D(fig)
	for label, group in data:
		ax.scatter(group[first_dimension], group[second_dimension], group[third_dimension], label = label)
	bm.plt.legend()
	bm.plt.show()

def analyze_feature_importance_by_plot(model_name, X_test, y_test, kind = 'separated'): # kind in ['separated', 'acumulative']
	"""Tries to analize how important features are by randomly modifing them.
	- model_name: model to use for the predictions.
	- kind: can be:
		- separated: the modified events are perforemd one at a time.
		- acumulative: the modified events are performed cumulatively.
	"""
	model = bm.keras.models.load_model('saved_models/' + model_name)
	results = []
	print('Evaluating data, this may take a while')
	for variable in range(0,X_test.shape[1]):
		print(variable)
		if kind == 'separated':
			X_values = bm.np.copy(X_test.values[:])
		elif kind == 'acumulative':
			X_values = X_test.values[:]
		X_values[:,variable] = bm.np.random.rand(X_values.shape[0])
		results.append(model.evaluate(X_values, y_test.values, verbose = 0)[1])
	if kind == 'separated':
		bm.plt.subplot(1,2,1)
		bm.plt.suptitle('Tissue prediction accuracy by randomly modified event.', fontsize = 25)
		bm.plt.plot(results)
		bm.plt.ylabel('Accuracy from 0 to 1', fontsize = 20)
		bm.plt.xlabel('Randomly modified exon', fontsize = 20)
		bm.plt.tick_params(labelsize = 15)
		bm.plt.ylim(0, 1)
		bm.plt.subplot(1,2,2)
		bm.plt.plot(results)
		bm.plt.xlabel('Randomly modified exon', fontsize = 20)
		bm.plt.ylabel('Accuracy', fontsize = 20)
		bm.plt.tick_params(labelsize = 15)
		bm.plt.show()
	elif kind == 'acumulative':
		bm.plt.title('Tissue prediction accuracy by acumulative randomly modified event.', fontsize = 25) 
		bm.plt.plot(results)
		bm.plt.ylabel('Accuracy', fontsize = 20)
		bm.plt.xlabel('Randomly modified exon', fontsize = 20)
		bm.plt.tick_params(labelsize = 15)
		bm.plt.show()
	return results

def print_separation_by_weight_and_tissue(data, labels, maxes, mines, tissue, dim1 = 'max', dim2 = 'min'):
	""" Prints how two exons separate a tissue
	maxes: the most correlated event to plot.
	mines: the most anti-correlated event to plot.
	tissue: from which tissue.
	dim1: whether to plot first the max or the min.
	dim2: whether to plot second the max or the min.
	"""
	dimensions = {'max':maxes, 'min':mines}
	if dim1 == dim2:
		second_index = 1
	else:
		second_index = 0
	first_exon = dimensions[dim1][tissue][0]
	second_exon = dimensions[dim2][tissue][second_index]
	bm.plt.scatter(data[first_exon], data[second_exon])
	bm.plt.scatter(data[first_exon][labels[tissue] == 1], data[second_exon][labels[tissue] == 1])
	bm.plt.xlabel(first_exon + ' inclusion', fontsize = 20)
	bm.plt.ylabel(second_exon + ' inclusion', fontsize = 20)
	bm.plt.show()

def build_predicting_tissue_model_dtc(X_train, y_train, X_CV, y_CV, X_test,y_test, max_depth = None):
	"""Builds a decision tree classifier that predicts the tissue of a sample
	-max_depth: the maximum depth of the decision tree classifier."""
	dtc = DecisionTreeClassifier(max_depth = max_depth)
	print('Constructing decision tree classifier')
	dtc.fit(pandas.concat([X_train, X_CV]), pandas.concat([y_train, y_CV]))
	print(dtc.score(X_test, y_test))
	return dtc

def analyse_tissue_prediction_from_NN(model, testdata, testlabels):
	"""Makes a report on which the accuracy per tissue is displayed for a model prediction."""
	numbers_to_tissues = {x:y for x,y in zip(range(len(testlabels.columns.unique())),testlabels.columns.unique())}
	predictions = model.predict_classes(testdata.values)
	predictions = [numbers_to_tissues[x] for x in predictions]
	real = testlabels.idxmax(1)
	print(classification_report(real, predictions))

## -- General pipelines

def general_model_build(n_samples = 10e10, n_variables = 10000, 
	model_name = 'tissue_classifier_filter_tissue_from_correct_', kind_of_data = 'psi', 
	filter_tissues = True, kind_of_model = 'NN', model_shape = 'simple', source = 'GTEX',
	label_data = 'Tissue', epochs = 1000): # kind_of_model in ['NN', 'DTC']
	"""General pipeline for the creation of a predicting tissue
	-n_samples: number of samples to use.
	-n_variables: number of variables to use.
	-model_name = file to store the data.
	-kind_of_data: kind of data to build the model: psi or gene.
	-filter tissue: whether to filter a tissue by a certain amount.
	-kind_of_model = NN for neural network or DTC for decision tree classifier.
	- model_shape=how to build the neueral network. check the networks.py file for details.
	-source: wether to read from GTEX or TCGA.
	- label_data: which variable to use as label data
	-epochs: how many epochs to train the model."""
	data, labels = read_psi_and_recover_tissue(n_samples = n_samples, n_variables = n_variables, 
		data_type = kind_of_data, filter_tissues = filter_tissues, source = source, 
		label_data = label_data)
	X_train, y_train, X_CV, y_CV, X_test,y_test = generate_sets(data, labels, norm = False)
	if kind_of_model == 'NN':
		model = build_predicting_tissue_model(X_train.values, y_train.values, X_CV.values, y_CV.values, 
			X_test.values,y_test.values, model_name + kind_of_data + '_' + str(n_variables),
			 model_shape = model_shape, epochs = epochs)
		return model
	elif kind_of_model == 'DTC':
		dtc = build_predicting_tissue_model_dtc(X_train, y_train, X_CV, y_CV, X_test, y_test)
		return dtc

def general_analyze_feature_importance(n_samples = 10e10, n_variables = 10000, 
	kind_of_data = 'psi', kind ='separated', filter_tissues = True, model_name = 'tissue_classifier_filter_tissue_from_correct_'):
	"""Peforms a pipeline for the analysis of the feature importance of a model"""
	data, labels = read_psi_and_recover_tissue(n_samples = n_samples, n_variables = n_variables, data_type = kind_of_data, filter_tissues = filter_tissues)
	X_test, y_test = generate_sets(data, labels, do_not_split = True)
	results = analyze_feature_importance_by_plot(model_name + kind_of_data + '_' + str(n_variables) , X_test, y_test, kind= kind)
	return results

def perform_MDS_analysis(n_samples = 10e10, n_variables = 10000, data_type = 'psi', filter_tissues = True, n_dimensions = 2, metric = True):
	""" Performs the MDS of the PSI/TPM values. It theoretically tries to capture the varibility of the data in non-linear ways"""
	data, labels = read_psi_and_recover_tissue(n_samples = n_samples, n_variables = 10000, data_type = data_type, filter_tissues = filter_tissues)
	X_train, y_train = generate_sets(data, labels, do_not_split = True)
	mds = MDS(n_components=n_dimensions, metric=metric, n_init=2, max_iter=1000, verbose=1, eps=0.0001, n_jobs=3, random_state=None, dissimilarity='euclidean')
	mds.fit(X_train.values)
	results = mds.embedding_
	results = pandas.DataFrame(results, columns = [str(x) + 'D' for x in range(1, n_dimensions+1)], index = y_train.index)
	results = pandas.concat([results, y_train.idxmax(1)], axis = 1)
	results = results.rename(columns = {0:'Tissue'})
	plot_by_group(results.groupby('Tissue'), '1D', '2D', kind_of_summary = 'MDS')

def perform_tSNE_analys(n_samples = 10e10, n_variables = 10000, data_type = 'psi', filter_tissues = True, n_dimensions = 2, perplexity = 30, learning_rate = 200, n_iter = 1000):
	""" Performs the tSNE of the PSI/TPM values. It is used to visualize high-dimensional data, converting affinities of data points to probabilities using t-Students distributions."""
	data, labels = read_psi_and_recover_tissue(n_samples = n_samples, n_variables = 10000, data_type = data_type, filter_tissues = filter_tissues)
	X_train, y_train = generate_sets(data, labels, do_not_split = True)
	tsne = TSNE(n_components=n_dimensions, perplexity= perplexity, early_exaggeration=12.0, learning_rate= learning_rate, n_iter= n_iter, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=1, random_state=None)
	tsne.fit(X_train.values)
	results = tsne.embedding_
	results = pandas.DataFrame(results, columns = [str(x) + 'D' for x in range(1, n_dimensions+1)], index = y_train.index)
	results = pandas.concat([results, y_train.idxmax(1)], axis = 1)
	results = results.rename(columns = {0:'Tissue'})
	plot_by_group(results.groupby('Tissue'), '1D', '2D')

def perform_feature_analysis_by_tissue(n_samples = 10e10, n_variables = 10000, data_type = 'psi', model = 'saved_models/tissue_classifier_filter_tissue_from_correct_psi_10000_single_layer', tissue = 'Brain', first_dimension = 'max', second_dimension = 'max'): 
	"""Extracts the importance on the first layer of a NN for each tissue"""
	data, labels = read_psi_and_recover_tissue(n_samples = n_samples, n_variables = n_variables, data_type = data_type, filter_tissues = True)
	X_test, y_test = generate_sets(data, labels, do_not_split = True)
	model = bm.keras.models.load_model(model)
	weights = extract_importance_from_single_layer_model_by_tissue(data, labels, model)
	m, n = extract_feature_importance_single_layer_NN(weights)
	print_separation_by_weight_and_tissue(data, labels, weights, m, n, tissue, first_dimension, second_dimension)
	return m,n, weights

def extract_exons_per_tissue_psis(kmers = 5, sections = 1, filter_by_psi = False):
	"""Tries to directly get the info from the files and build a model thas has as 
	target the psis for all the tissues at the same time."""
	data = read_psis(n_cols = 8000, nrows = 30000, transpose = True, drop_na = True) 
	data.columns = data.columns.str.split('.').str[0]
	exons = pandas.read_table('exons_sequence_200_40_40_200.txt', names = ['Exon_ID', 'Exon_Seq', 'Length'], index_col = 'Exon_ID')
	data = recover_tissue_information(data, True, 'GTEX')
	means = data.groupby('_primary_site').mean()
	stds = data.groupby('_primary_site').std()
	counts = data.groupby('_primary_site').count()
	means = means.transpose()
	means = means.drop('<not provided>', axis = 1)
	means['Seq'] = exons.loc[means.index]['Exon_Seq']
	means = means.dropna()
	if kmers != 0: # ==0 means Don't generate kmers
		kmers = sequences_to_kmers(means['Seq'],ks = kmers, sections = sections)
		means.drop('Seq', axis = 1, inplace = True)
		means = means.join(kmers)
	labels = means[means.columns[:31]]
	means = means.drop(means.columns[:31], axis = 1)
	if filter_by_psi:
		means = means[(labels.mean(1) < filter_by_psi)|(labels.mean(1) > 1- filter_by_psi)]
		labels = labels[(labels.mean(1) < filter_by_psi)|(labels.mean(1) > 1-filter_by_psi)]
	return means, labels

def compare_prediction_with_metadata():
	"""Tries to infer how important is the metadata and how it affects the prediction by means of violinplots."""
	data, labels = read_psi_and_recover_tissue(10e10, 40000, data_type = 'psi', source  = 'GTEX')
	model = keras.models.load_model('saved_models/tissue_prediction_GTEX')
	proba = model.predict_proba(data.values)
	proba = pandas.DataFrame(proba, index = data.index, columns = labels.columns)
	proba = proba.max(1)
	proba.index = proba.index.str.split('.').str[0:2].str.join('-')
	proba.name = 'Max Softmax Output'
	metadata = pandas.read_table('gtex_pheno/GTEx_v7_Annotations_SubjectPhenotypesDS.txt', index_col = 0)
	metadata = metadata.join(pandas.DataFrame(proba), how = 'inner')
	metadata = metadata.dropna()
	seaborn.violinplot(y = 'Max Softmax Output', x = 'AGE', data = metadata, cut = 0, order = sorted(metadata['AGE'].unique()))
	del data, labels

def build_model_from_direct_files_multiple_label(sections = 1 ,epochs = 1000, 
	learning_rate = 0.001, layers = 3, nodes = 100):
	"""Does a multilabel prediction from a file."""
	data, labels = extract_exons_per_tissue_psis(kmers = 5, sections= sections)
	model = networks.plain_NN(data.shape[1], labels.shape[1], layers, nodes, 
		step_activation = 'relu', final_activation = 'sigmoid',optimizer = False, 
		kind_of_model = 'classification', halve_each_layer = True,dropout = True)
	print(model.summary())
	history = networks.fit_network(model, data, labels.round())
	return history

def best_tissue_prediction_model(model_name, data_type = 'psi', n_features = 10000000, batch_size = 30, use_PCA = False):
	"""Performs the best pipeline we found forthe prediction of the tissue. 
	- use_PCA: uses first a PCA to summarize the data and reduce dimensionality."""
	data, labels = read_psi_and_recover_tissue(500, n_features, data_type = data_type, source  = 'GTEX')
	if use_PCA:
		data = pandas.DataFrame(PCA(n_components = 12000).fit_transform(data), index = data.index)
	X_train, X_cvt, y_train, y_cvt = train_test_split(data, labels, train_size = 0.80)
	X_CV, X_test, y_CV, y_test = train_test_split(X_cvt, y_cvt, train_size = 0.50)
	print('Variables used: '+ str(X_CV.shape[1]))
	model = Sequential()
	model.add(Dense(300, activation='relu', input_dim=X_CV.shape[1]))
	model.add(Dropout(0.5))
	model.add(Dense(150, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(60, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(32, activation='softmax'))
	model.summary()
	print(X_CV.shape[1])
	sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False) #Default optimizer
	model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
	checkpointer = ModelCheckpoint(filepath='saved_models/' + model_name,  verbose=1, save_best_only=True)
	history = model.fit(X_train.values, y_train.values, batch_size=batch_size, epochs=1000,validation_data=(X_CV.values, y_CV.values), verbose=1, callbacks = [checkpointer])
	model.load_weights('saved_models/' + model_name)
	print(model.evaluate(X_test.values, y_test.values))
	return model, X_test, y_test


#model_name = 'model_40k_gene' +str(random.random()) 
#best_tissue_prediction_model(model_name, data_type = 'gene', batch_size = 30, n_features = 1000000)
