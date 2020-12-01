episis leei na exoume ekpedeumena montela kai gia ta 2 erwtimata

na doume an getParameters idio kai sta dio wste n bei utils

sto autoencoder menei mono na ftiaxnoume gia apantisi 2 ta diagrammata pou zitaei sin na doume uperparametrous lelelle

kati parksena warnings to autoencoder alla mpourdes den epireazoun

ta arxeia mas den einai .dat

baloume sta requirments to hiplot

2o erwtima zitaei h5 to file ara mallon prepei etsi

usage of autoencoder:
	python autoencoder + path to file with input data
	for example :

		python autoencoder.py data/t10k-images-idx3-ubyte
	
	then enter the hyperparameter the programm asks you to 

usage of classification:
python classification.py -d data/train-images-idx3-ubyte --d1 data/t10k-images-idx3-ubyte -t data/train-labels-idx1-ubyte --t1 data/t10k-labels-idx1-ubyte --model models/autoencoder_softmax_sigmoid/
	
	
autoencoder parameters : 
	ari8mos sineliktikwn strwmatwn
	mege8ow sineliktinwn filtrwn
	ari8mos sineliktikwn filtrwn ana strwma
	epochs
	batch_size
	pli8os neurwnwn sto fc layer
