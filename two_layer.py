from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np


#input_data
mnist = input_data.read_data_sets("mnist",one_hot=True)

#file_name
file_name = "mnist/model.ckpt"
#Parameters
batch_size = 128 #batch size
epochs = 20 #number of iterations
learning_rate = 0.01 #learning rate of the NN
display_step = 1

n_input = 784 #number of input features
n_classes = 10 #number of output

n_hidden_layer = 256 #hidden layer size/width

#weights and biases
weights = {
	'hidden_layer': tf.Variable(tf.random_normal([n_input,n_hidden_layer]),name='weights_hidden'),
	'output_layer': tf.Variable(tf.random_normal([n_hidden_layer,n_classes]),name='weights_output')
}

biases = {
	'hidden_layer': tf.Variable(tf.zeros([n_hidden_layer]),name='biases_hidden'),
	'output_layer': tf.Variable(tf.zeros([n_classes]),name='biases_output')
}

#
# x = tf.placeholder(tf.float32, [None,28,28,1])
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None, n_classes])

# x_flat = tf.reshape(x, [-1, n_input])

#hidden layer with linear activation
layer_1 = tf.add(tf.matmul(x,weights['hidden_layer']), biases['hidden_layer'])

#hidden layer with relu activation
layer_1 = tf.nn.relu(layer_1)

#output layer with linear activation
logits = tf.add(tf.matmul(layer_1,weights['output_layer']), biases['output_layer'])

#loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)

	for i in range(epochs):
		total_batch = int(mnist.train.num_examples/batch_size)
		for j in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)

			sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})

		if(i%10==0):
			valid_accuracy = sess.run(accuracy,feed_dict={x:mnist.validation.images, y:mnist.validation.labels})
			print('Epoch {:<3} - Validation Accuracy: {}'.format(i, valid_accuracy))

	saver.save(sess, file_name)
	print("train model saved")