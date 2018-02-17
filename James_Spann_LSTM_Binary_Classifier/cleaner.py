"""
cleaner.py by James Spann

This file does all of the heavy lifting in terms of cleaning the data for use.

All of the organizing dataset by tweet index and cleaning of the JSON files is done here.
The documentation for this file is in README.md
"""
import sys
import numpy as np
import pandas as pd
import csv
import random
import json
import re
from random import shuffle

SMALLER1YEARLOC = "./smaller1year.json"
COMBINEDSMALLERLOC = "../combined_annotationsSMALLER.json"
gwords = None
numFeatures = 25	# for 25 dimensional vectors



########################################

def loadGloveData():
	"""
	Uses pandas to get the word embeddings from the glove dataset
	"""
	global gwords
	gloveFilename = "../glove.twitter.27B/glove.twitter.27B."+str(numFeatures)+"d.txt"
	gwords = pd.read_table(gloveFilename, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
	# gwords = gwordsTable.as_matrix().astype('float32')
	# return gwords.as_matrix().astype('float32')

def getTwitterIDClasses():
	totalIDs = {}

	with open(COMBINEDSMALLERLOC,"rb") as myfile:
		for line in myfile:
			tweetData = json.loads(line.decode('utf8'))
			totalIDs[tweetData["tweet_id"]] = tweetData["topic_human"]
	return totalIDs

def gloveindiciesUnknown():
	"""
	 This function takes 'smaller1year.json' (the file with "tweetID" and "tweetText") in SMALLER1YEARLOC
	and 'combined_annotationsSMALLER.json' (the file with "topic_human" and "tweet_id"),
	and converts each word in the tweet and
	"""
	tweetIDClassez = getTwitterIDClasses()
	miffp = open("modelIndexFile.json","wb")#The file to which we write cleaned rows

	largestTextLen = 0 #Longest number of words in a tweet
	numincorrect = 0 #The number of tweets that we saw that we got correct
	testCase = 0
	with open(SMALLER1YEARLOC,"rb") as myfile:
		for line in myfile:
			tweet = json.loads(line.decode('utf8'))
			myText = tweet['tweetText'].lower()
			splitText = myText.split(" ")
			lineToWrite = ""
			for element in splitText:
				# print(element)
				if element == "":
					pass #We don't add empty spaces to our tweet line
				else:
					# e = element
					#TODO: Do a replace for the double quotes in the JSON tweet
					try:
						# if the word is not found in the word embeddings, we append <unknown>
						currIndex = gwords.index.get_loc(element) #Looks in the pandas frame for the wordEmbedding
						lineToWrite = lineToWrite + element + " "
						# print
					except Exception as e:
						# raise e
						numincorrect = numincorrect + 1
						lineToWrite = lineToWrite + "<unknown> "
					finally:
						pass
			testCase = testCase + 1
			if (testCase % 1000) == 0:
				print("[Update] We are on the "+str(testCase)+"th tweet")
			# miffp.write(b"{\"tweetID\":\""+str(testCase).encode()+b"\n")
			miffp.write(b"{\"tweetID\":\""+str(tweet['tweetID']).encode()+b"\", \"tweetText\":\" "+str(lineToWrite).encode()+b"\", \"tweetClass\":\""+str(tweetIDClassez[tweet['tweetID']]).encode()+b"\"}\n")
				#{"tweetID":"476075004531318784","tweetText":"Hungry as a bitch at work tho"}

	myfile.close()
	miffp.close()
	print("\nCOMPLETE!")
	# print(str(numincorrect)+" words were marked with '<unknown>'.")
	print("In the file loaddata.py (in the tensorflow model), you need to update the parameter 'max_encoder_time' to be: "+str(largestTextLen)+".")
	return

def gloveindiciesUnknownSplit(testRate=0.5):
	"""
	 This function takes 'smaller1year.json' (the file with "tweetID" and "tweetText") in SMALLER1YEARLOC
	and 'combined_annotationsSMALLER.json' (the file with "topic_human" and "tweet_id"),
	and converts each word in the tweet and

	the testRate is the percentage of how we want to split our data. For example: testRate=0.8 means 80% of the data is for testing and 20% is for training
	"""
	tweetIDClassez = getTwitterIDClasses()
	miffp = open("modelIndexFile.json","wb")
	miffpTest = open("modelIndexFileTest.json","wb")

	largestTextLen = 0 #Longest number of words in a tweet
	numincorrect = 0 #The number of tweets that we saw that we got correct
	testCase = 0

	quickOpen = open(SMALLER1YEARLOC,"rb")
	mlines = quickOpen.readlines()
	shuffle(mlines)
	quickOpen.close()

	testJob = 0#how many of our training sets are Job related
	testNotJob = 0#how many of our training sets are not Job related
	testNegYield = int((len(mlines) * testRate)/2)
	testPosYield = int((len(mlines) * testRate)/2)

	with open(SMALLER1YEARLOC,"rb") as myfile:
		for line in mlines:
			tweet = json.loads(line.decode('utf8'))
			myText = tweet['tweetText'].lower()
			splitText = myText.split(" ")
			lineToWrite = ""
			for element in splitText:
				# print(element)
				if element == "":
					pass #We don't add empty spaces to our tweet line
				else:
					# e = element
					#TODO: Do a replace for the double quotes in the JSON tweet
					try:
						currIndex = gwords.index.get_loc(element) #Looks in the pandas frame for the wordEmbedding
						lineToWrite = lineToWrite + element + " "
						# print
					except Exception as e:
						# raise e
						numincorrect = numincorrect + 1
						lineToWrite = lineToWrite + "<unknown> "
					finally:
						pass
			testCase = testCase + 1
			if (testCase % 1000) == 0:
				print("[Update] We are on the "+str(testCase)+"th tweet")
			# miffp.write(b"{\"tweetID\":\""+str(testCase).encode()+b"\n")
			if (testNotJob < testNegYield) and tweetIDClassez[tweet['tweetID']] == "notjob":
				miffpTest.write(b"{\"tweetID\":\""+str(tweet['tweetID']).encode()+b"\", \"tweetText\":\" "+str(lineToWrite).encode()+b"\", \"tweetClass\":\""+str(tweetIDClassez[tweet['tweetID']]).encode()+b"\"}\n")
				testNotJob = testNotJob + 1

			elif (testJob < testPosYield) and tweetIDClassez[tweet['tweetID']] == "job":
				miffpTest.write(b"{\"tweetID\":\""+str(tweet['tweetID']).encode()+b"\", \"tweetText\":\" "+str(lineToWrite).encode()+b"\", \"tweetClass\":\""+str(tweetIDClassez[tweet['tweetID']]).encode()+b"\"}\n")
				testJob = testJob + 1

			else:
				miffp.write(b"{\"tweetID\":\""+str(tweet['tweetID']).encode()+b"\", \"tweetText\":\" "+str(lineToWrite).encode()+b"\", \"tweetClass\":\""+str(tweetIDClassez[tweet['tweetID']]).encode()+b"\"}\n")
				#{"tweetID":"476075004531318784","tweetText":"Hungry as a bitch at work tho"}

	myfile.close()
	miffp.close()
	print("\nCOMPLETE!")
	# print(str(numincorrect)+" words were marked with '<unknown>'.")
	print("In the file loaddata.py (in the tensorflow model), you need to update the parameter 'max_encoder_time' to be: "+str(largestTextLen)+".")
	return

def gloveindicies():
	return

if __name__ == '__main__':
	if len(sys.argv) > 1:
		loadGloveData()
		if sys.argv[1] == "completeIndicies":
			#We want to convert the file to indicies for glove embeddings
			gloveindicies()
		elif sys.argv[1] == "unknownIndicies":
			#We want to convert the file to indicies for glove embeddings with with the '<unknown>' tag
			gloveindiciesUnknown()
		elif sys.argv[1] == "unknownIndiciesSplit":
			#We want to convert the file to indicies for glove embeddings with with the '<unknown>' tag AND split it for testing and training
			gloveindiciesUnknownSplit()
	else:
		print("Usage: 'python3 cleaner.py <argument>' ")
