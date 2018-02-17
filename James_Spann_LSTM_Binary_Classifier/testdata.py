import tensorflow as tf
import loaddata as ld


def f1score(tfsess,logitsVariable,totalTweetData):
	"""
	tfsess - tensorflow session
	"""
	tp = 0 #True positive
	tn = 0 #True negative
	fp = 0 #False positive
	fn = 0 #False negative

	### Running through the training sets
	for currtweetID in totalTweetData.keys():
		enpInp, decInp, declen, decOut, targetWeight, tweetClass = ld.getTweetForTraining(currtweetID)	
		
		testOutput = tfsess.run(logitsVariable,feed_dict={encoder_inputs: enpInp, decoder_inputs: decInp, decoder_lengths: declen, decoder_outputs: decOut })
		testdog = tfsess.run(getFinalSoftmax(testOutput))

		modelOutput = classifyDecodedOutput(testdog)
		print("The model said"+" ".join(modelOutput))
		if modelOutput == "off on":#model said - notjob
			if tweetClass == "notjob":#correct output was notjob - true negative
				tn = tn + 1
			else:#correct output was job - false negative
				fn = fn + 1
		else:#model said - job
			if tweetClass == "notjob":#correct output was notjob - false positive
				fp = fp + 1
			else:#correct output was job - true positive
				tp = tp + 1
		# print("Key update: "+)

	### Running the math
	precision = (tp / (tp + fp))
	recall = (tp / (tp + fn))
	calculatedValue = (2 / ((1 / recall) + (1 / precision)))

	print("Precision value: "+str(precision))
	print("Recall value: "+str(recall))
	print("Full F1 value: "+str(calculatedValue))

	return calculatedValue