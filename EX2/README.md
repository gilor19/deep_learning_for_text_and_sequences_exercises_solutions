In this exercise, I implemented three types of two-layer perceptron-based POS/NER taggers.

The architecture includes an embedding layer for the input features. The three taggers vary by the type of features representing the data:

Tagger1 - the features are words.
Tagger3 - the features are words, their prefixes, and suffixes.
Tagger4 - the features are words and their characters.

Tagger1 and Tagger3 implementations support both random initialization and pre-trained vectors for the initialization of the word embedding. Tagger4 implementation supports only pre-trained vectors for the initialization of the word embedding.


##########################################################

All taggers require the following arguments:

--train <path_to_data_file> --dev <path_to_labels_file> --test <path_to_output_file> -lr <learning_rate_value> -e <number_of_epochs>

##########################################################

0. A prerequisite is to run pip install -r requirements.txt

1. To run the code for NER task, add --ner argument to rum configurations. This will adjust the accuracy computation.

2. To run the code with pre-trained vectors, you need to have the following files in your working directory:
	    - wordVectors.txt  (the loaded vectors file)
	    - vocab.txt        (the loaded vocab file)
	
	and to add --load as argument.
    

3. After running the code, the following files will be created in your working directory:

	a. accuracy.png - a plot of the dev accuracy scores by epochs.
	b. loss.png - a plot of the dev and test losses by epochs.
	c. test_preds - the prediction of the test (if test file was given).


4. The test path argument is optional.


