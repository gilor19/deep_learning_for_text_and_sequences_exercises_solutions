In this exercise I implemented a BiLSTM based POS/NER tagger using PyTorch.

The tagger supports 4 different representations of the input data (features to input into the network):
	a. words
	b. characters
	c. prefixes and suffixes
	d. combination of both words and charcters
	

The arguments for running the training proccess:

 <representation_type(a-d)> <path_to_train_file> <path_to_save_model> <path_to_save_vocab_file> [OPTIONAL --dev <path_to_dev_file>] [OPTIONAL --ner] 
 
 
The arguments for running the predition proccess:
<representation_type(a-d)> <path_to_test_file> <path_to_saved_model> <path_to_saved_vocab_file> <path_to_save_prediction_file> 


