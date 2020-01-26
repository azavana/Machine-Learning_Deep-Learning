# Spliting data function
import numpy as np

def split_dataset(x_dataset, y_dataset, ratio): # Takes the input and output dataset as well as the desired split ratio

	# Shuffles a list of numbers
	arr = np.arange(x_dataset.size)
	np.random.shuffle(arr)
	num_train = int(ratio*x_dataset.size) # Calculates the number of training examples
	
	# Uses the shuffled llist to split the x_dataset and y_dataset
	x_train = x_dataset[arr[0:num_train]]
	x_test = x_dataset[arr[num_train:x_dataset.size]]
	y_train = y_dataset[arr[0:num_train]]
	y_test = y_dataset[arr[num_train:x_dataset.size]]
	
	# Returns the split x and y datasets
	return x_train, x_test, y_train, y_test
