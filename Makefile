run-autoencoder:
	python autoencoder.py data/t10k-images-idx3-ubyte

run-classification:
	python classification.py -d data/train-images-idx3-ubyte --d1 data/t10k-images-idx3-ubyte -t data/train-labels-idx1-ubyte --t1 data/t10k-labels-idx1-ubyte --model models/autoencoder_linear_128_20/