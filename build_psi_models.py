import pandas
import build_vcf_models as bm
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import networks
from sklearn.metrics import classification_report
## -- Reading and preparaing

def read_psis(n_cols = 7000, nrows = 1000, transpose = True, drop_na = True, source = 'GTEX'):
	print('Reading PSI data')
	if source == 'GTEX':
		data = pandas.read_table('correct_psis.txt', usecols = range(0, min(7862, n_cols)), nrows =  min(37706,nrows))
		data = data.loc[~data.index.duplicated(keep= 'first')]
	if drop_na:
		data = data.dropna(1)
	if transpose:
		data = data.transpose()
	return data

def read_TPM(n_cols = 10e10, nrows = 1000, transpose = False, drop_na = True, source = 'TCGA'): # data in ['GTEX','TCGA']
	print('Reading TPM data of isoforms')
	if source == 'GTEX':
		data = pandas.read_table('/genomics/users/aruiz/psis/GTPX_TPM_transpose.txt', usecols = range(0, n_cols), nrows = nrows, index_col = 0)
	elif source == 'TCGA':
		data = pandas.read_table('/genomics/users/aruiz/TCGA/TcgaTargetGtex_rsem_isoform_tpm', usecols = range(0, nrows), nrows = n_cols, index_col = 0)
		data = 2**data - 0.001
		data = data.transpose()
	data = data.reindex_axis(sorted(data.columns), axis=1)
	if drop_na:
		data = data.dropna(1)	
	if transpose:
		data = data.transpose()
	return data

def recover_sequence_from_exons(exons):
	exons = pandas.read_table('exons_sequence_200_40_40_200.txt', names = ['Exon_ID', 'Exon_Seq', 'Length'], index_col = 'Exon_ID')



def recover_tissue_information(data, substitute = True, source = 'GTEX'): # source in ['GTEX', 'TCGA']
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
	tissues = data.groupby('_primary_site')['_primary_site'].count()
	tissues = tissues[tissues > minim]
	data[data['_primary_site'].isin(tissues)]
	data = data[data['_primary_site'].isin(tissues.index)]
	return data

def read_psi_and_recover_tissue(n_samples, n_variables = 10000, data_type = 'psi', 
filter_tissues = True,ensure_all_tissues = True, source = 'GTEX', label_data = 'Tissue'): #data_type in ['psi', 'tpm']; tpm_data in ['GTEX','TCGA'] label_data in ['Tumor', 'Tissue']
	if data_type == 'psi':
		data = read_psis(nrows = n_variables, n_cols = n_samples, source = source)
	elif data_type == 'tpm':
		data = read_TPM(nrows = n_samples, n_cols = n_variables, source = source)
	if source == 'GTEX':
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
	if ensure_all_tissues and filter_tissues:
		if source == 'GTEX':
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
	print('generating sets')
	if norm:
		data = bm.normalize(data)
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

def get_PCA_data(pca, labels, data):
	projected = pca.transform(data)
	projected = bm.pandas.DataFrame(projected, columns = ['PC'+ str(x) for x in list(range(1, projected.shape[1]+1))])
	projected['labels'] = labels.values
	pca_data = projected.groupby('labels')
	return pca_data


def build_pca(n_samples = 10e10, n_variables = 10000, data_type = 'psi', filter_tissues = True,
	ensure_all_tissues = True):
	data, labels = read_psi_and_recover_tissue(n_samples = n_samples, n_variables = n_variables, data_type = data_type, filter_tissues = filter_tissues,ensure_all_tissues = ensure_all_tissues)
	print('Computing PCA')
	pca = PCA()
	pca.fit(data)
	return pca

def extract_importance_from_pca(pca, data, PC):
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

def build_predicting_tissue_model(X_train, y_train, X_CV, y_CV, X_test,y_test,model_name = 'tissue_classifier_from_correct_psis', model_shape = 'simple', batch_size = 256):
	print('Building model')
	model_file = 'saved_models/' + model_name + model_shape
	if model_shape == 'simple':
		model = bm.create_simple_NN(X_CV.shape[1], y_CV.shape[1], activation = 'softmax')
	if model_shape == 'single_layer':
		model = bm.single_layer_NN(X_CV.shape[1], y_CV.shape[1], activation = 'softmax')
	sgd = bm.keras.optimizers.SGD(lr=0.0001, decay= 0, momentum=0.9, nesterov=True)
	model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	checkpointer = bm.ModelCheckpoint(filepath=model_file,  verbose=0, save_best_only=True)
	model.fit(X_train, y_train, batch_size=batch_size, epochs=15000,validation_data=(X_CV, y_CV), verbose=1, callbacks = [checkpointer])
	model.load_weights(model_file)
	model.save(model_file)
	model.evaluate(X_test, y_test)
	

def plot_by_group(data,first_dimension, second_dimension, kind_of_summary = 'tSNE'): # kind_of_summary in ['PCA', 'MDS', 'tSNE']
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
	fig = bm.plt.figure()
	ax = Axes3D(fig)
	for label, group in data:
		ax.scatter(group[first_dimension], group[second_dimension], group[third_dimension], label = label)
	bm.plt.legend()
	bm.plt.show()

def analyze_feature_importance_by_plot(model_name, X_test, y_test, kind = 'separated'): # kind in ['separated', 'acumulative']
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



def print_separation_by_weight_and_tissue(data, labels, weights, maxes, mines, tissue, dim1 = 'max', dim2 = 'min'):
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
	dtc = DecisionTreeClassifier(max_depth = max_depth)
	print('Constructing decision tree classifier')
	dtc.fit(pandas.concat([X_train, X_CV]), pandas.concat([y_train, y_CV]))
	print(dtc.score(X_test, y_test))
	return dtc

def analyse_tissue_prediction_from_NN(model, testdata, testlabels):
	numbers_to_tissues = {x:y for x,y in zip(range(len(labels.columns.unique())),labels.columns.unique())}
	predictions = model.predict_classes(testdata.values)
	predictions = [numbers_to_tissues[x] for x in predictions]
	real = testlabels.idxmax(1)
	print(classification_report(predictions, real))

## -- General pipelines

def general_model_build(n_samples = 10e10, n_variables = 10000, 
	model_name = 'tissue_classifier_filter_tissue_from_correct_', kind_of_data = 'psi', 
	filter_tissues = True, kind_of_model = 'NN', model_shape = 'simple', source = 'GTEX',
	label_data = 'Tissue'): # kind_of_model in ['NN', 'DTC']
	data, labels = read_psi_and_recover_tissue(n_samples = n_samples, n_variables = n_variables, 
		data_type = kind_of_data, filter_tissues = filter_tissues, source = source, 
		label_data = label_data)
	X_train, y_train, X_CV, y_CV, X_test,y_test = generate_sets(data, labels)
	if kind_of_model == 'NN':
		build_predicting_tissue_model(X_train.values, y_train.values, X_CV.values, y_CV.values, X_test.values,y_test.values, model_name + kind_of_data + '_' + str(n_variables), model_shape = model_shape)
	elif kind_of_model == 'DTC':
		dtc = build_predicting_tissue_model_dtc(X_train, y_train, X_CV, y_CV, X_test, y_test)
		return dtc

def general_analyze_feature_importance(n_samples = 10e10, n_variables = 10000, 
	kind_of_data = 'psi', kind ='separated', filter_tissues = True, model_name = 'tissue_classifier_filter_tissue_from_correct_'):
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

def build_model_from_direct_files_multiple_label(sections = 1 ,epochs = 1000, 
	learning_rate = 0.001, layers = 3, nodes = 100):
	data, labels = extract_exons_per_tissue_psis(kmers = 5, sections= sections)
	model = networks.plain_NN(data.shape[1], labels.shape[1], layers, nodes, 
		step_activation = 'relu', final_activation = 'sigmoid',optimizer = False, 
		kind_of_model = 'classification', halve_each_layer = True,dropout = True)
	print(model.summary())
	history = networks.fit_network(model, data, labels.round())
	return history