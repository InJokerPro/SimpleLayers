import tensorflow as tf 

def Convolutional2D(input,stacks,sampler_size,stride=1,max_pooling=True,stddev=0.1,activation = 'relu',padding='SAME',name=None) : 
	input_shape = input.get_shape()

	if not input_shape : 
		print('Need to input a 4D Tensor')
		
	w = tf.Variable(tf.truncated_normal(shape=[sampler_size,sampler_size,int(input_shape[3]) ,stacks],stddev=0.1),name=name+'_weights')

	b = tf.Variable(tf.constant(0.1,shape=[stacks]),name=name+'_biases')
	f = tf.nn.conv2d(input=input, filter=w, strides=[1,1,1,1], padding=padding) + b
	
	if activation is None : 
		r = f
	else :
		d = {
			'relu' : tf.nn.relu(f),
			'sigmoid' : tf.nn.sigmoid(f),
			'softplus' : tf.nn.softplus(f),
		}

		r = d[activation]

	if max_pooling : 
		return tf.nn.max_pool(r, ksize=[1,2,2,1], strides=[1,2,2,1], padding=padding)
	else :
		return r

def FullyConnected(input,units,activation='relu',name=None,keep_prob=0.8) : 
	input_shape = input.get_shape()

	# Alpha code, in progress
	input_dim=1
	for i in input_shape[1:] : 
		input_dim = input_dim*int(i)

	input = tf.reshape(input, shape=[-1,input_dim])
	w = tf.Variable(tf.truncated_normal(shape=[input_dim,units],stddev=0.1),name=name+'_weights')
	b = tf.Variable(tf.constant(0.1,shape=[units]),name=name+'_biases')
	f = tf.matmul(input, w) + b 
	if activation is None : 
		r = f
	else :
		d = {
			'relu' : tf.nn.relu(f),
			'sigmoid' : tf.nn.sigmoid(f),
			'softplus' : tf.nn.softplus(f),
		}

		r = d[activation]

	r = tf.nn.dropout(r, keep_prob=tf.Variable(tf.constant(keep_prob)))

	return r