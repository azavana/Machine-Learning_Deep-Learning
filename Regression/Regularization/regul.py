from split_data import split_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
training_epochs = 1000
reg_lambda = 0.
x_dataset = np.linspace(-1, 1, 50)
num_coeffs = 9
y_dataset_params = [0.] * num_coeffs
y_dataset_params[2] = 1
y_dataset = 0
for i in range(num_coeffs):
	y_dataset += y_dataset_params[i] * np.power(x_dataset, i)
y_dataset += np.random.randn(*x_dataset.shape) * 0.3

(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.7) # Splits the dataset into 70% training and 30% testing
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

def model(X, w):
	terms = []
	for i in range(num_coeffs):
		term = tf.multiply(w[i], tf.pow(X, i))
		terms.append(term)
	return tf.add_n(terms)

# Splits the dataset into 70% training and 30% testing
w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)
cost = tf.div(tf.add(tf.reduce_sum(tf.square(Y-y_model)),tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w)))),2*x_train.size)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Sets up the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Tries various regularization parameters
for reg_lambda in np.linspace(0,1,50):
	for epoch in range(training_epochs):
		sess.run(train_op, feed_dict={X: x_train, Y: y_train})
	final_cost = sess.run(cost, feed_dict={X: x_test, Y:y_test})
	print('reg lambda', reg_lambda)
	print('final cost', final_cost)
	
sess.close()

