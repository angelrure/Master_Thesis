import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

def analyze_posterior_binary_classifier(probs, labels, model_name = 'Random Forest'):
	"""Given a model, a data and the labels of the data, predicts how well the prediction 
	probabilities are.""" 
	""" working on that
	global temporal_info, accuracies
	data = pandas.DataFrame(probs, columns = ['probs'], index = labels.index)
	data['labels'] = labels
	accuracies = {}
	for i in range(50,101, 1):
		temporal_info = data[data['probs']>=i/100]
		accuracies[i] = accuracy_score(data['probs'].round(), data['labels'])
	return accuracies
	"""
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
	pca = PCA(n_components = 100)
	pca.fit(data)
	data = pandas.DataFrame(pca.transform(data), index = data.index)
	return data

def perform_leave_on_out(data):
	loo = LeaveOneOut()
	loo = loo.split(data)
	return loo

def data_parser(data, reference, outfile):
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

## CONCRETE WORKLINES

def TCGA_isoTPM_to_geneTPM():
	data = pandas.read_table('/genomics/users/aruiz/TCGA/TcgaTargetGtex_rsem_isoform_tpm', 
		index_col = 0)
	data = data.transpose()
	data.columns = data.columns.str.split('.').str[0]
	data = 2**(data-0.001)
	reference = pandas.read_table('transcripts_per_gente.txt', index_col = 0)
	data_parser(data, reference, '/genomics/users/aruiz/TCGA/TcgaTargetGtex_gene_isoform_tpm')
