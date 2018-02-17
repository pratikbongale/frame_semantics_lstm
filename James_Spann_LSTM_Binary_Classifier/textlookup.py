""""
This script is used to look at the incoming data an pull necessary values out of it
"""
import numpy as np
import pandas as pd
import csv
import random
import json


gwords = None
tweetCaptions = {} #Key: id of a tweet, Value: text of tweet
tweetClasses = {} #Key: Tweet ID, Value: classification of a tweet ("job" or "not job")


def getVocabSize():
	"""
	Returns: int with number of unique words in total across all tweets
	"""
	return

def getEmbeddingsSize():
	return

def getEmbeddingKeys():
	"""
	Returns a list of keys with all the words in the embeddings
	"""
	# "../glove.twitter.27B/glove.twitter.27B.25d"
	return


mydictionary = {}

def rewriteTweetFile():
	finalFile = open("smaller1year.json","wb")

	annotatedTweets = 0
	with open("../combined_annotationsSMALLER.json", "r") as fp:
		for line in fp:
			jstr = json.loads(line)
			mydictionary[str(jstr['tweet_id'])] = jstr['topic_human']
			annotatedTweets = annotatedTweets + 1
			if annotatedTweets % 1000000 == 0:
				print("combined_annotationsSMALLER.json - We are on:"+str(annotatedTweets))

	print("Switching to the next one")

	simpleKeys = mydictionary.keys()
	totalTweets = 0
	with open("../new1year.json", "r") as fp2:
		for line in fp2:
			jstr = json.loads(line)
			if str(jstr['id']) in simpleKeys:
				# print "Foundone!"
				lastData = "{\"tweetID\":\""+str(jstr['id'])+"\",\"tweetText\":"+jstr['text']+"\"}"
				json.dump(lastData, finalFile)
				finalFile.write("\n")
			else:
				pass
			'''
			try:
				mydictionary[str(jstr['id'])] #Will fail if the value is not in the dictionary
				lastData = "{\"tweetID\":\""+str(jstr['id'])+"\",\"class\":\""+mydictionary[str(jstr['id'])]+"\",\"tweetText\":"+jstr['text']+"\"}"
				json.dump(lastData, finalFile)
				finalFile.write("\n")
			except Exception as e:
				# raise e
				print "error: ",e
			finally:
				pass
			'''
			totalTweets = totalTweets + 1
			if totalTweets % 1000000 == 0:
				print("new1year.json - We are on:"+str(totalTweets))

	fp2.close()
	fp.close()
	finalFile.close()

	print("annotatedTweets: "+str(annotatedTweets))
	print("totalTweets: "+str(totalTweets))
	print("Wow we actually made it!")
	return


def getLongestTweet():
	annotatedTweets = 0
	maxLen = 0
	with open("smaller1year.json", "r") as fp:
		for line in fp:
			jstr = json.loads(line)
			print(str(jstr['tweetText']).split(" "))
			splitsize = len(str(jstr['tweetText']).split(" "))
			if (splitsize > maxLen):
				maxLen = splitsize
			# mydictionary[str()] = jstr['topic_human']
			# annotatedTweets = annotatedTweets + 1
			if annotatedTweets % 1000000 == 0:
				print("smaller1year.json - We are on:"+str(annotatedTweets))

	return splitsize

tweetIndex = [] #A list of words
tweetVectors = None


####FROM OTHER LOAD DATA FILE
def loadGloveData(numFeatures):
	"""
	Uses pandas to get the word embeddings from the glove dataset
	"""
	global gwords
	gloveFilename = "./../glove.twitter.27B/glove.twitter.27B."+str(numFeatures)+"d.txt"
	gwords = pd.read_table(gloveFilename, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
	gwords = gwords.as_matrix().astype('float32')


def loadTweetClasses():
	global tweetClasses
	# tweetClasses = {}
	annotatedTweets = 0
	with open("../combined_annotationsSMALLER.json", "r") as fileTweetTexts:
		for line in fileTweetTexts:
			jstr = json.loads(line)
			tweetClasses[str(jstr['tweet_id'])] = jstr['topic_human']
			# print(str(jstr['tweet_id']))
			annotatedTweets = annotatedTweets + 1
			if annotatedTweets % 1000000 == 0:
				print("combined_annotationsSMALLER.json - We are on:"+str(annotatedTweets))
	# return tweetClasses

def loadTweetTexts():
	global tweetCaptions
	with open('../new1year.json', 'r') as jsonfile:
		for line in jsonfile:
			tweet = json.loads(line)
			myText = tweet['text']
			tweet_id = tweet['id']
			# print myText
			tweetCaptions[tweet_id] = myText
	return tweetCaptions

# loadGloveData(25)
# print(type(gwords))
# print(gwords[0])
# rewriteTweetFile()

# print("Largest:"+str(getLongestTweet())) #This is where I got max size 17