import tensorflow as tf
import numpy as np
import sys
from tensorflow.python.layers import core as layers_core

from loaddata import *


batch_size = 1

num_units = 25

max_gradient_norm = 1

learning_rate = 0.02

# max_encoder_time = 50	# Maximum length of the encoder input - Double check this
max_encoder_time = 42	# The code said that the longest was 40 so I added some padding
max_decoder_time = 2	# Maximum length of the encoder input - Double check this

epochs = 1

if len(sys.argv) > 1 and (sys.argv[1] == "train"):
	##Loads the necessary file with our tweets and prints out the F1 Score
	setupInitialData("modelIndexFile.json")#Must be done first

elif len(sys.argv) > 1 and (sys.argv[1] == "trainterm"):
	##Does everything "train" does and gives a term to test user input sentences
	setupInitialData("modelIndexFile.json")#Must be done first
	global continueTermPrompt #variable we use to say we want to continue into the prompt after training
	continueTermPrompt = True

elif len(sys.argv) > 1 and (sys.argv[1] == "play"):
	##Trains against my testfile (Probably not in the repo)
	setupInitialData("playfultweets.json")#Must be done first
	global continueTermPrompt #variable we use to say we want to continue into the prompt
	continueTermPrompt = True
else:
	#Just don't do it
	print("You have given me no commands. I need something to do.\nWhy not try something like this: 'python3 main.py train'.")
	sys.exit(0)
	# setupInitialData(None)#Must be done first



def f1score(tfsess,logitsVariable,totalTweetData):
	"""
	tfsess - tensorflow session
	"""
	tp = 0 #True positive
	tn = 0 #True negative
	fp = 0 #False positive
	fn = 0 #False negative

	### Running through the training sets
	# for currtweetID in totalTweetData.keys():
	# resetInputFile()
	resetInputFileForTraining("modelIndexFileTest.json")
	for r in range(getTotalLines()-1):
		enpInp, decInp, declen, decOut, targetWeight, tweetClass = loadNextTrainingTweet()
		
		testOutput = tfsess.run(logitsVariable,feed_dict={encoder_inputs: enpInp, decoder_inputs: decInp, decoder_lengths: declen, decoder_outputs: decOut })
		testdog = tfsess.run(getFinalSoftmax(testOutput))

		modelOutput = classifyDecodedOutput(testdog)
		modelOutput = " ".join(modelOutput)
		print("The model said \'"+modelOutput+"\'")
		if modelOutput == "off on":#model said - notjob
			if tweetClass == "notjob":#correct output was notjob - true negative
				tn = tn + 1
			else:#correct output was job - false negative
				fn = fn + 1
		else:#model said - job ("on off")
			if tweetClass == "notjob":#correct output was notjob - false positive
				fp = fp + 1
			else:#correct output was job - true positive
				tp = tp + 1
		print("Key update: tn-",tn," fn-",fn," fp-",fp," tp-",tp)

	### Running the math
	precision = (tp / (tp + fp))
	recall = (tp / (tp + fn))
	calculatedValue = (2 / ((1 / recall) + (1 / precision)))

	print("Precision value: "+str(precision))
	print("Recall value: "+str(recall))
	print("Full F1 value: "+str(calculatedValue))

	return calculatedValue



def classifyDecodedOutput(decodedOutput):
	"""
	The input shape should be (batchSize, max_decoder_time, vocabSize)
	"""

	print("max decode time:",max_decoder_time)
	finalValues = [] #The resulting length will have the same number of values as the input decode_input
	for x in range(max_decoder_time):
		print("teh decode time, this time is:",x)
		finalValues.append(getDecodedOutputFromIndex(np.argmax(decodedOutput[0][x])))
	# print finalValues
	return finalValues

def getFinalSoftmax(rnnOutputArray):
	"""
	Gets an rnnOutputArray which should be a numpy array
	and runs it in a tf softmax function
	"""
	return tf.nn.softmax(rnnOutputArray)

### The start of the Tensorflow model
with tf.variable_scope("embedding") as scope:
	##Load glove embeddings matrix
	embedding_encoder = tf.Variable(getEmbeddings())

	##Load the embeddings for the encoder sentence
	encoder_inputs = tf.placeholder(tf.int32, shape=[max_encoder_time, batch_size], name="encoderInput")
	encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)

	##Load the embeddings for the decoder sentence
	decoder_inputs = tf.placeholder(tf.int32, shape=[max_decoder_time, batch_size], name="decoderInput")
	decoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, decoder_inputs)

	decoder_outputs = tf.placeholder(tf.int32, name="decoderOutput")


##Dense layer
with tf.variable_scope("dense") as denseScope:
	projection_layer = layers_core.Dense(getVocabSize(), use_bias=False) #TODO: change when we add jobsEmbedding matrix


##Encoder
with tf.variable_scope("encoder") as encoderScope:
	# Build RNN cell
	encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

	# This is unnecessary but can be added as a parameter for the dynamic_rnn for verifyability
	#source_sequence_length = tf.Variable(batch_size,dtype=tf.int32)

	# defining initial state
	#encoder_outputs is a tensor with the shape (max_encoder_time, batchSize, num_units)
	#encoder_state is a LSTMStateTuple
	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_emb_inp, time_major=True, dtype=tf.float64)


##Decoder
with tf.variable_scope("decoder") as decoderScope:
	decoder_lengths = tf.placeholder(tf.int32, shape=[1])#The sequence length -  An int32 vector tensor.

	##Build RNN cell
	decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
	
	##Helper
	helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True)

	##Decoder
	decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)

	##Dynamic decoding
	outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True,maximum_iterations=3)
	logits = outputs.rnn_output # (batchSize, output length (should be 2), vocabulary size aka glove embedding (should be 1193514))


##Calculating loss
with tf.variable_scope("loss") as lossScope:
	crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)#Returns a tensor with shape of decoder_outputs

	target_weights = tf.placeholder(tf.float64)#, shape=[1])
	train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)


# Calculate and clip gradients
with tf.variable_scope("gradient") as gradScope:
	params = tf.trainable_variables()
	gradients = tf.gradients(train_loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)


##Optimization
with tf.variable_scope("optimizer") as optimScope:
	optimizer = tf.train.AdamOptimizer(learning_rate)
	update_step = optimizer.apply_gradients(zip(clipped_gradients, params))


with tf.Session() as sess:
	print(sess.run(tf.global_variables_initializer()))

	c = 0
	while c < epochs:
		# enpInp, decInp, declen, decOut, targetWeight = getWord()

		# currentLoss = sess.run(train_loss,feed_dict={encoder_inputs:enpInp, decoder_inputs:decInp, decoder_lengths:declen, decoder_outputs:decOut, target_weights:targetWeight }) 
		# print(currentLoss)
		# sess.run(update_step,feed_dict={encoder_inputs:enpInp, decoder_inputs:decInp, decoder_lengths:declen, decoder_outputs:decOut, target_weights:targetWeight })

		#Works: enpInp, decInp, declen, decOut, targetWeight = getTweetForTraining(getRandomTree())
		for r in range(getTotalLines()-1):
			enpInp, decInp, declen, decOut, targetWeight = loadNextTweet()
			currentLoss = sess.run(train_loss,feed_dict={encoder_inputs:enpInp, decoder_inputs:decInp, decoder_lengths:declen, decoder_outputs:decOut, target_weights:targetWeight }) 
			print(currentLoss)
			sess.run(update_step,feed_dict={encoder_inputs:enpInp, decoder_inputs:decInp, decoder_lengths:declen, decoder_outputs:decOut, target_weights:targetWeight })

			if currentLoss < 2:
				learning_rate = 0.0001
			c = c + 1
		resetInputFile()

	#Test
	print("Model trained!")

	# enpInp, decInp, declen, decOut, targetWeight = getWord()	
	# testOutput = sess.run(logits,feed_dict={encoder_inputs:enpInp, decoder_inputs:decInp, decoder_lengths:declen, decoder_outputs:decOut, target_weights:targetWeight })
	# testdog = sess.run(getFinalSoftmax(testOutput))

	# print(testdog)
	# print(testdog.shape)
	# print(type(testdog))

	# translate = classifyDecodedOutput(testdog)
	# print(translate)
	print("======-------------======")
	print("Going to calculate F1 score:")
	f1score(tfsess=sess, logitsVariable=logits, totalTweetData=twttrData)

	if continueTermPrompt == True:
		theUserInput = None
		while theUserInput != "":
			theUserInput = input("&")

			sanity, enpInp, decInp, declen, decOut, targetWeight = parseSentenceForDebugging(theUserInput)
			if sanity == -1:
				pass#Do not continue because the input is not valid
			else:
				testOutput = sess.run(logits,feed_dict={encoder_inputs: enpInp, decoder_inputs: decInp, decoder_lengths: declen, decoder_outputs: decOut })

				# testOutput = sess.run(logits,feed_dict={encoder_inputs: enpInp})
				testdog = sess.run(getFinalSoftmax(testOutput))

				mout = classifyDecodedOutput(testdog)
				print("The model said: \""+str(mout)+"\" ")
