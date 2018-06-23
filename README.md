# Master Thesis

It contains the scripts used for the analysis of the Master's project. They are:
* build_psi_models: contains the main functionalities used for the creation of the tissue prediction and the PSI prediction models. 
It also contains several useful functions to analyse data.
* build_vcf_models_5.0: contains different functions for the analysis of the currently working on VCF model.
* create_psis_model_sequence: easily creates a model to predict the PSI values based on sequence. It is better to use the 
build_psi_models when possible, as it is more clean and complete.
* create_psis_model_sequence: easily creates a model to predict the PSI values based on sequence and conservations. It is better to use the 
build_psi_models when possible, as it is more clean and complete.
* networks: contains different functions to easily create, fit and analyse neural networks models. 
* utilities: contains several utilities. Like analyzing posterior probabilities, create roc curves, confusion matrices, PCAs and the
in-silico study we did in the manuscript.
