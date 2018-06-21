import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import itertools

# This provides several handy functions used.


def analyze_posterior_binary_classifier(probs, labels, model_name = 'Random Forest'):
	"""Given a model, a data and the labels of the data, predicts how well the prediction 
	probabilities are.""" 
	if len(labels.unique()) == 2:
		information = pandas.DataFrame(index = labels.index)
		information['label'] = labels
		try:
			information['predicted_probabilities'] = model.predict_proba(data)[:,1]
		except:
			information['predicted_probabilities'] = model.predict_proba(data)
		information = information[information['predicted_probabilities'] >= 0.5]
		accuracies_total = {}
		accuracies_relative = {}
		samples_remaining = {}
		for i in range(50,101, 1):
			temporal_info = information[information.predicted_probabilities > i/100]
			if temporal_info.shape[0] == 0:
				break
			accuracies_relative[i] = (temporal_info[(temporal_info.label == temporal_info.predicted_probabilities.round())].shape[0])
			accuracies_total[i] = accuracies_relative[i]/information.shape[0]
			samples_remaining[i] = temporal_info.shape[0]/information.shape[0]
			try:
				accuracies_relative[i] = accuracies_relative[i]/temporal_info.shape[0]
			except ZeroDivisionError:
				pass
	print(accuracies_relative)
	print(accuracies_total)
	pandas.Series(accuracies_relative).plot(label = 'Relative accuracy', fontsize = 20)
	pandas.Series(accuracies_total).plot(label = 'Total accuarcy', fontsize = 20)
	pandas.Series(samples_remaining).plot(label = 'Percentage of samples remaining', fontsize = 20)
	plt.xlabel("Model's certainty in %", fontsize =  20)
	plt.ylabel('Accuracy', fontsize = 20)
	plt.title('Accuracy of ' + model_name +' model by model\'s certainty for the prediction of ' + labels.name, fontsize = 20)
	plt.legend(fontsize = 20)
	plt.show()

def perform_roc_curve(labels, scores, reason = ''):
	"""Performs the ROC curve analysis.
	labels = real labels.
	scores = predicted probabilities for the labels.
	reason = the reason of the ROC. It will continue the phrase: ROC curve  ... """
	fpr, tpr, thr = roc_curve(y_true = labels, y_score = scores, pos_label = 1)
	plt.plot(fpr, tpr, label = 'ROC curve (AUC = ' + str(auc(fpr, tpr).round(2))  + ' )')
	plt.plot([0,1], [0,1])
	plt.xlabel('False positive rate', fontsize = 20)
	plt.xticks(fontsize = 20)
	plt.ylabel('True positive rate', fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.title('Reciever operating characteristic curve ' + reason, fontsize = 20)
	plt.legend(fontsize = 20, loc = 4)
	plt.show()

def perform_prc_curve(labels, scores, reason = ''):
	"""Performs the precision-recall curve analysis.
	labels = real labels.
	scores = predicted probabilities for the labels.
	reason = the reason of the ROC. It will continue the phrase: Precision-recall curve  ... """
	precision, recall, _ = precision_recall_curve(labels, scores)
	plt.step(recall, precision, color='b', where='post', label = 'Average precision: '+ str(average_precision_score(labels, scores).round(3)))
	plt.xlabel('Recall', fontsize = 20)
	plt.ylabel('Precision', fontsize = 20)
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.yticks(fontsize = 20)
	plt.xticks(fontsize = 20)
	plt.title('Precision-recall curve ' + reason, fontsize = 20)
	plt.legend(fontsize = 20, loc = 4)
	plt.show()

def perform_confusion_matrix(y_predicted, y_true):
	"""Performs a confusion matrix given a set of predicted and true labels."""
	numbers_to_tissues = {x:y for x,y in zip(range(len(y_true.columns.unique())),y_true.columns.unique())}
	predictions = [numbers_to_tissues[x] for x in y_predicted]
	real = y_true.idxmax(1)
	return (confusion_matrix(real, predictions))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """Plots a confusion matrix object.
    -classes: names of the classes to use, if not given it is inferred from the cm.
    -normalize: whether to normalize the values from the confusion matrix by class.
    -cmap: color map object to use to color the matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_results_from_several_file(common_name):
	"""Takes a set of files with a common name and plots them together. Example: results_method1, results_method2, etc."""
	list_of_files = [x for x in os.listdir() if common_name in x]
	list_of_files.remove(common_name+'.txt')
	data_frames = []
	for file in list_of_files:
		name = file.split(common_name)[1][1:-4]
		data = pandas.read_table(os.getcwd()+'/'+file, sep = ',', names = ['Drug', name+'_MAE', name+'_R2'], index_col = 'Drug', usecols = [name+'_R2', 'Drug'])
		data = data[~data.index.duplicated(keep = 'first')]
		data_frames.append(data)
	for i in range(1, len(data_frames)):
		data_frames[0] = data_frames[0].join(data_frames[i])
	data = data_frames[0]
	data = data.dropna()
	data.sort_values('TPMIsoforms_R2', ascending = False).plot(fontsize = 20)
	plt.ylabel('Determination coefficient (R2)', fontsize = 20)
	plt.legend(fontsize = 20)
	plt.title('Random Forest Determination Coefficient of IC prediction including isoforms', fontsize = 20)
	plt.show()


def perform_pca(data, n_components):
	"""Performs a simple pca"""
	pca = PCA(n_components = 100)
	pca.fit(data)
	data = pandas.DataFrame(pca.transform(data), index = data.index)
	return data

def perform_leave_on_out(data):
	"""Performs a simple leave one out method"""
	loo = LeaveOneOut()
	loo = loo.split(data)
	return loo

def data_parser(data, reference, outfile):
	"""Parses a data using a reference.
	-reference: reference to parse the data.
	-outfile: where to store the parsed data"""
	reference = reference.sort_index()
	parsed = pandas.DataFrame(index = data.index)
	max_length = len(reference.index.unique())
	i = 0
	for gene in reference.index.unique():
		print('\r Percentage completion: ' + str(round(i/max_length,3)) + '%', end = '')
		i = i + 1
		transcripts = reference.loc[gene].values
		transcripts = [x[0] for x in transcripts]
		try:
			parsed[gene] = data[transcripts].sum(1) 
		except:
			continue
	parsed.to_csv(outfile)

def analyze_pca_results_labels(data, pca, labels):
	"""Plots the labels of a data within the first two dimensions of a PCA."""
	tags = labels.idxmax(1)
	transformed_data = pandas.DataFrame(pca.transform(data),index = data.index)
	for tissue in tags.unique():
		plt.scatter(transformed_data.loc[tags[tags == tissue].index][0], 
			transformed_data.loc[tags[tags == tissue].index][1], label = tissue)
	plt.legend()
	plt.show()

## CONCRETE WORKLINES

def TCGA_isoTPM_to_geneTPM():
	"""Parses data from TCGA"""
	data = pandas.read_table('/genomics/users/aruiz/TCGA/TcgaTargetGtex_rsem_isoform_tpm', 
		index_col = 0)
	data = data.transpose()
	data.columns = data.columns.str.split('.').str[0]
	data = 2**(data-0.001)
	reference = pandas.read_table('transcripts_per_gente.txt', index_col = 0)
	data_parser(data, reference, '/genomics/users/aruiz/TCGA/TcgaTargetGtex_gene_isoform_tpm')

### Fix paper

def analyze_tissue_pca_similarity(data, labels, pca, tissues):
	"""To study whether the tissues that were worstly predicted present similar
	splicing patterns."""
	tags = labels.idxmax(1)
	data = data[tags.isin(['Breast','Salivary Gland', 'Uterus', 'Vagina', 'Pancreas', 'Spleen'])]
	labels = labels[tags.isin(['Breast','Salivary Gland', 'Uterus', 'Vagina', 'Pancreas', 'Spleen'])]
	analyze_pca_results_labels(data, pca, labels)

def in_silico_screening(model_name, kind_of_model, n_samples, 
	use_base_instead_of_mutated = False, n_exons_to_use = 2, separate_tissues = False,
	include_conservation = True, keep_base_sequence = True, include_len = False,
	include_Tissue = False):
	"""Performs an in-silico mutation analysis in which a sequence is mutated to check how these 
	mutations affect the PSI values.
	-model_name = file to use for making the predictions.
	-kind_of_model: whether using a classification or regression model.
	-n_exon_to_use = number of exon to use and plot in the in silico mutation.
	-separate_tissues: wther to perform the analysis separating the samples by tissue.
	-include_conservation: whether to use the conservation also in the models.
	-keep_base_sequence: whether to keep the base sequence in the plots.
	-include_len: whether to use the len as a variable for the predictions.
	-include_tissue: whether the model needs the tissue for predictions.
	"""
	data = pre_process_data(kind_of_model = 'Classification',n_samples =n_samples, 
		skip_rows = 0, use_base_instead_of_mutated =  True, compute_kmers = False,
		include_conservation = include_conservation)
	if separate_tissues == False:
		data = data[~data.index.duplicated(keep = 'first')]
	if include_len == False:
		data.drop('len', inplace = True, axis = 1, errors = 'ignore')
	if include_Tissue == False:
		data.drop('Tissue', inplace = True, axis = 1)
	modified_sequences = []
	for index, row in data.iterrows():
		for position in range(1,481):
			modified_sequences.append(change_sequence_and_conservation_values(row, position, False, 
				include_conservation = include_conservation))
	data = pandas.DataFrame(modified_sequences)
	data = bm.calculate_kmers(data, keep_base_sequence = keep_base_sequence, 
		verify_nucleotides = True)
	model = bm.keras.models.load_model(model_name)
	X_train, y_train = bm.generate_sets(data, do_not_split = True, avoid_overlap = False, 
		use_tissues = include_Tissue, using_kmers = True, norm = True, validate_tissues = True, 
		train_set_size = 0.8, get_output_in_pandas_format = True, filter_by_number = 0)
	predictions = model.predict(X_train.values)
	predictions = predictions.reshape(-1, 480)
	for prediction in predictions:
		plot_prediction(prediction, 'Norm')
	plt.show()
	return predictions

def analyze_tissue_tsne_similarity(data, labels, ignore_tissues = ['Breast','Salivary Gland', 'Uterus', 'Vagina', 'Pancreas', 'Spleen']):
	"""To study whether the tissues that were worstly predicted present similar
	splicing patterns.
	-ignore tissues: tissues to avoid plot.
	"""
	tags = labels.idxmax(1)
	data = data[~tags.isin(ignore_tissues)]
	labels = tags[~tags.isin(ignore_tissues)]
	pca = PCA(n_components = 100)
	pca.fit(data)
	print(pca.explained_variance_ratio_.sum())
	transformed_data = pandas.DataFrame(pca.transform(data),index = data.index)
	tsne = TSNE(method = 'exact')
	transformed_data =  pandas.DataFrame(tsne.fit_transform(X = transformed_data),index = transformed_data.index)
	for tissue in labels.unique():
		print(tissue)
		tissue_transformed_data = transformed_data[labels == tissue]
		plt.scatter(tissue_transformed_data[0], tissue_transformed_data[1], label = tissue)
	plt.legend()
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.title(str(len(ignore_tissues)) + ' worse predicted tissues t-SNE', fontsize = 20)
	plt.show()
