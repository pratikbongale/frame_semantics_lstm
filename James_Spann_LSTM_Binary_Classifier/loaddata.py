import numpy as np
import pandas as pd
import csv
import random
import json
import re
from collections import defaultdict

gwords = None #Glove embeddings matrix
twttrData = None # key:[class,text]
 
embeddingSize = 25 #Number of features in the word embedding

max_encoder_time = 42#Maximum length of the encoder input - Double check this
max_decoder_time = 2#Maximum length of the encoder input - Double check this

####Model Data
def getWord():
	firstSentence = "I am a cat and dog"
	encoderIndicies = sentenceToIndicies(cleanSentence(firstSentence))
	decoderIndicies = getDecoderIndiciesForTweetKey("355032504710868994")

	# print("encoder indexes:",encoderIndicies)
	# print("decoder stuff:",decoderIndicies)


	# targw = getTargetWeights(2)
	# print(""+len(encoderIndicies))
	return encoderIndicies, decoderIndicies, np.array([2]), decoderIndicies, getTargetWeights(2)


def getTweetForTraining(tweetID):
	firstSentence = twttrData[tweetID][1]#"I am a cat and dog"
	encoderIndicies = sentenceToIndicies(cleanSentence(firstSentence))
	decoderIndicies = getDecoderIndiciesForTweetKey(tweetID)

	decoderLengths = 2#Since we only write 2 words
	tweetOutputClass = twttrData[tweetID][0]
	return encoderIndicies, decoderIndicies, np.array([decoderLengths]), decoderIndicies, getTargetWeights(decoderLengths), tweetOutputClass

def parseSentenceForDebugging(userSentence):
	"""
	This is the function used in the prompt at the end of the program. It converts a sentence to the necessary format
	"""

	# global mylastTweetID
	# global maintweetJSON
	userSentence = userSentence.lower()

	# print("CurrTweet ("+str(mylastTweetID)+"): "+maintweetJSON[mylastTweetID]["tweetText"])
	# maintweetJSON[mylastTweetID]["tweetText"]
	# maintweetJSON[mylastTweetID]["tweetClass"]
	vawit = verifyAllWordsInTweet(userSentence)
	if vawit != True:#True if it works. False if it doesnt
		print("Hey! You gave us a word we can't use! ["+str(vawit)+"]")
		return -1, 0, 0, 0, 0, 0

	encoderIndicies = sentenceToIndicies(userSentence)#NOTE: This is never used in the model. I just need it as a placeholder
	decoderIndicies = getDecoderIndiciesForClass("notjob")#NOTE: This is never used in the model. I just need it as a placeholder

	decoderLengths = 2#Since we only write 2 words
	# tweetOutputClass = maintweetJSON[mylastTweetID]["tweetClass"]

	# mylastTweetID = mylastTweetID + 1
	return 1, encoderIndicies, decoderIndicies, np.array([decoderLengths]), decoderIndicies, getTargetWeights(decoderLengths)

def loadNextTrainingTweet():
	"""

	I copied and pasted this from 'loadNextTweet'
	This is the function used to load tweets from 'modelIndexFile.json' along with its classification.
	The tweets are formatted so that that they are read line by line
	"""

	global mylastTweetID
	global maintweetJSON

	print("CurrTweet ("+str(mylastTweetID)+"): "+maintweetJSON[mylastTweetID]["tweetText"])
	# maintweetJSON[mylastTweetID]["tweetText"]
	# maintweetJSON[mylastTweetID]["tweetClass"]
	encoderIndicies = sentenceToIndicies(cleanSentence(maintweetJSON[mylastTweetID]["tweetText"]))
	decoderIndicies = getDecoderIndiciesForClass(maintweetJSON[mylastTweetID]["tweetClass"])

	decoderLengths = 2#Since we only write 2 words
	tweetOutputClass = maintweetJSON[mylastTweetID]["tweetClass"]

	mylastTweetID = mylastTweetID + 1
	return encoderIndicies, decoderIndicies, np.array([decoderLengths]), decoderIndicies, getTargetWeights(decoderLengths), tweetOutputClass

def loadNextTweet():
	"""
	This is the function used to load tweets from 'modelIndexFile.json'.
	The tweets are formatted so that that they are read line by line
	"""
	# firstSentence = twttrData[tweetID][1]#"I am a cat and dog"
	global mylastTweetID
	global maintweetJSON

	print("CurrTweet ("+str(mylastTweetID)+"): "+maintweetJSON[mylastTweetID]["tweetText"])
	# maintweetJSON[mylastTweetID]["tweetText"]
	# maintweetJSON[mylastTweetID]["tweetClass"]
	encoderIndicies = sentenceToIndicies(cleanSentence(maintweetJSON[mylastTweetID]["tweetText"]))
	decoderIndicies = getDecoderIndiciesForClass(maintweetJSON[mylastTweetID]["tweetClass"])

	decoderLengths = 2#Since we only write 2 words
	tweetOutputClass = maintweetJSON[mylastTweetID]["tweetClass"]

	mylastTweetID = mylastTweetID + 1
	return encoderIndicies, decoderIndicies, np.array([decoderLengths]), decoderIndicies, getTargetWeights(decoderLengths)

####Helper functions
def sentenceToIndicies(sentence):
	"""
	Precondition: Sentence has been cleaned

	Converts a sentence to a list of indicies
	"""
	finalSentence = sentence.split(" ")

	wordsIndicies = []
	for currword in finalSentence:
		if (currword == "") or(currword == " "):
			continue

		try:
			currIndex = glovewords.index.get_loc(currword) #Looks in the pandas frame for the wordEmbedding
			
		except Exception as e:
			# raise e
			# print("You've got an error: |"+currword)
			currIndex = 410057 #Predetermined value that isn't used in our test data. The value is "<unknown>"
		finally:
			wordsIndicies.append(currIndex)

	
	
	# np.append(np.zeros(shape=[13,1]), np.ones(shape=[2,1])).reshape(15,1)
	# np.append(np.zeros(shape=[13,1]), np.ones(shape=[2,1])).reshape(15,1)

	#Add zeros so we can pass our sentence
	# print(wordsIndicies)
	# print(str(max_encoder_time-len(wordsIndicies)))
	# senIndiciesArray = np.asarray(wordsIndicies).reshape(len(finalSentence),1)
	senIndiciesArray = np.asarray(wordsIndicies).reshape(len(wordsIndicies),1)
	finalArray = np.append(senIndiciesArray,  np.zeros(shape=[max_encoder_time-len(wordsIndicies),1])  ).reshape(max_encoder_time,1)
	
	return finalArray


def getTargetWeights(indiciesWithValue):
	"""
	Creates a [1 x maxSize] numpy array where the values are either 1.0 or 0.0 .
	The values correspond to padded values(0.0) and real values (1.0) in the decoder_outputs
	tensor so that when the array gets multiplied by crossent later, the padded values won't
	be counted in the final result.
	"""
	returnable = np.zeros(max_decoder_time)
	for i in range(indiciesWithValue):
		returnable[i] = 1.0
	return returnable


def cleanSentence(sentence):
	newSentence = sentence

	##Separate Unicode values (\u...)

	##Replace Unicode values (\u...)
	unicodeSymbolsLower = re.findall(r"\\u[a-zA-Z0-9]*", newSentence)
	unicodeSymbols = re.findall(r"\\U[a-zA-Z0-9]*", newSentence) + unicodeSymbolsLower
	for sym in unicodeSymbols:
		newSentence = newSentence.replace(sym,"")#TODO: Fix this

	##Replace Mentions (@...)
	mentions = re.findall(r"@([a-zA-Z0-9_]{1,15})", newSentence)
	for mention in mentions:
		newSentence = newSentence.replace("@"+mention," <user> ")


	##Replace Hashtags (#...)
	hashtags = re.findall(r"#(\w+)", newSentence)
	for htag in hashtags:
		newSentence = newSentence.replace("#"+htag," <hashtag> ")

	##Replace urls (http://...)
	# urls = re.findall(r"(http?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)", newSentence)
	urls = re.findall(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", newSentence)
	for website in urls:
		# print(website)
		caught = "".join(website)
		# print(caught)
		newSentence = newSentence.replace(caught," <url> ")

	##Replace time
	times = re.findall(r"[\d]{1,2}:[\d]{1,2}", newSentence)
	for time in times:
		# print("Found time:",time)
		newSentence = newSentence.replace(time," <time> ")	

	##Replace Numbers
	numbers = re.findall(r"\b\d+\b", newSentence)
	for num in numbers:
		newSentence = newSentence.replace(num," <number> ")

	##Replace Contractions
	contractions = re.findall(r"\'ve", newSentence)
	for contract in contractions:
		newSentence = newSentence.replace(contract," \'ve")#We add a space because it is a glove embedding

	##Replace repeating values
	
	# repeating = re.findall(r"([!?.]){2,}", newSentence)
	repeating = re.findall(r"([!?.]){2,}", newSentence)
	for rep in repeating:
		newSentence = newSentence.replace(rep," <repeat> ")#We add a space because it is a glove embedding

	##Replace elongated words (waaaaayyyyyy)
	'''
	repeating = re.findall(r"(\S*?)(.)\2{2,}", newSentence)
	for rep in repeating:
		print(rep)
		newSentence = newSentence.replace(rep," <elong> ")#We add a space because it is a glove embedding
	'''



	# print(newSentence)

	newSentence = newSentence.replace("☺️","<smile>")
	newSentence = newSentence.replace("😓","")
	newSentence = newSentence.replace("n't"," n't ")
	newSentence = newSentence.replace("/"," / ")
	newSentence = newSentence.replace("?"," ? ")
	newSentence = newSentence.replace("."," . ")
	newSentence = newSentence.replace(","," , ")
	newSentence = newSentence.replace(":"," : ")
	newSentence = newSentence.replace(")"," ) ")
	newSentence = newSentence.replace("("," ( ")
	newSentence = newSentence.replace("/"," / ")
	newSentence = newSentence.replace("\""," \" ")
	# print(newSentence)
	return newSentence.lower()

def getDecoderIndiciesForTweetKey(tweetID):
	classification = twttrData[tweetID][0]
	if classification == "notjob":
		#47 - on
		#232 - off
		#For when the tweet is not about a job (0,1)
		return np.array([[232],[47]])
	else:
		#For when the tweet is about a job (1,0)
		return np.array([[47],[232]])

def getDecoderIndiciesForClass(classification):
	if classification == "notjob":
		#47 - on
		#232 - off
		#For when the tweet is not about a job (0,1)
		return np.array([[232],[47]])
	else:
		#For when the tweet is about a job (1,0)
		return np.array([[47],[232]])

def verifyAllWordsInTweet(tweetttt):
	try:
		elem = None
		for element in tweetttt.split(" "):
			if element == "":
				pass
			else:
				elem = element
				currIndex = glovewords.index.get_loc(element) #Looks in the pandas frame for the wordEmbedding

	except Exception as e:
		# raise e
		return elem
	finally:
		pass

	return True

####File data
def setupInitialData(mainFileStr):
	"""
	Loads glove data into 'gwords' and twitter data with its keys and classes into 'twttrData'
	"""

	loadGloveData()

	if mainFileStr == None: #we have not been provided a main tweet file

		twttrClasses = loadTweetClasses()
		twttrTexts = loadTweetTexts()
		# print(twttrClasses)
		# print(twttrTexts)

		global twttrData
		dd = defaultdict(list)

		for d in (twttrClasses, twttrTexts): # you can list as many input dicts as you want here
			# for key, value in d.iteritems():
			for key, value in d.items():
				dd[key].append(value)

		twttrData = dd # key:[class,text]
	else:#We were provided a file with all the the twitter information that we wanted! hurah!
		global maintweetFile
		global maintweetLines
		global maintweetJSON
		global mylastTweetID
		global myInpFileStr #The string of the file we are reading in

		myInpFileStr = mainFileStr

		maintweetFile = open(myInpFileStr,"r")
		maintweetLines = maintweetFile.readlines()
		maintweetJSON = []
		for l in maintweetLines:
			maintweetJSON.append(json.loads(l))

		mylastTweetID = 0#The index for which tweet we are looking at
	return

def loadGloveData():
	"""
	Uses pandas to get the word embeddings from the glove dataset
	The shape should be: (1193514, 25)
	"""
	global gwords

	global glovewords
	glovewords = None

	gloveFilename = ".././glove.twitter.27B/glove.twitter.27B."+str(embeddingSize)+"d.txt"
	glovewords = pd.read_table(gloveFilename, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
	gwords = glovewords.as_matrix().astype('float64')
	print("Glove Data loaded!")
	return

def loadTweetClasses():
	annotatedTweets = 0
	tweetClasses = {} #Key: Tweet ID, Value: classification of a tweet ("job" or "not job")
	with open("../combined_annotationsSMALLER.json", "r") as fileTweetTexts:
		for line in fileTweetTexts:
			jstr = json.loads(line)
			tweetClasses[str(jstr['tweet_id'])] = jstr['topic_human']
			# print(str(jstr['tweet_id']))
			annotatedTweets = annotatedTweets + 1
			if annotatedTweets % 1000000 == 0:
				print("combined_annotationsSMALLER.json - We are on:"+str(annotatedTweets))
	return tweetClasses

def loadTweetTexts():
	tweetCaptions = {} #Key: id of a tweet, Value: text of tweet
	with open("../smaller1year.json", "r") as jsonfile:
		for line in jsonfile:
			tweet = json.loads(line)
			myText = tweet['tweetText']
			tweet_id = tweet['tweetID']
			# print myText
			tweetCaptions[tweet_id] = myText
	return tweetCaptions

def getTwitterData():
	global twttrData
	return twttrData

def getEmbeddings():
	"""
	Returns glove embeddings
	"""
	global gwords
	return gwords

def getVocabSize():
	global gwords
	return gwords.shape[0]

def getJobEmbeddings():
	"""
	This returns the previously generated embeddings for the job data
	"""
	return np.load("jobEmbeddings.npy")

def getDecodedOutputFromIndex(idx):
	"""
	Takes a value that was decoded from the tensorflow model and
	returns the word associated with that index
	"""
	return glovewords.index[idx-1]

def getRandomTree():
	return random.choice(twttrData.keys())

def getTotalLines():
	"""
	This function is a compliment to 'loadNextTweet' in that it represents the
	number of times that function can be called before the program breaks
	"""
	global maintweetLines
	return len(maintweetLines)

def resetInputFile():
	global maintweetFile
	global maintweetLines
	global maintweetJSON
	global mylastTweetID
	global myInpFileStr #The string name of the file we are updating

	maintweetFile.close()

	maintweetFile = open(myInpFileStr,"r")
	maintweetLines = maintweetFile.readlines()
	maintweetJSON = []
	for l in maintweetLines:
		maintweetJSON.append(json.loads(l))

	mylastTweetID = 0#The index for which tweet we are looking at
	return

def resetInputFileForTraining(testFileStr):
	"""
	This is used when we are calculating our F1 score. We provide new data for the model so we can test things accordingly
	"""
	global maintweetFile
	global maintweetLines
	global maintweetJSON
	global mylastTweetID
	global myInpFileStr #The string name of the file we are updating

	maintweetFile.close()#assuming it is open
	
	myInpFileStr = testFileStr#assign it to the new test file

	maintweetFile = open(myInpFileStr,"r")
	maintweetLines = maintweetFile.readlines()
	maintweetJSON = []
	for l in maintweetLines:
		maintweetJSON.append(json.loads(l))

	mylastTweetID = 0#The index for which tweet we are looking at
	return
# setupInitialData()
# print(sentenceToIndicies(cleanSentence("J.P. Morgan #OpenSource #Job: CIB TECH – Equities – Java Developer – Associ... ( #NY , New York) http://t.co/WNYquJ4JqG #VeteranJob").lower()))
# print(sentenceToIndicies(cleanSentence("Someone please entertain me at work. PLEASE. #snapchat #texting #idecwhat")))
# print(sentenceToIndicies(cleanSentence("Love your work @emeticmedic - GR8 opportunity to add @PaleyStudios's \"Soliloquy\" http://jspann.me into the series: http://t.co/k712smJaKG")))
# print(glovewords.index.get_loc(currword) #Looks in the pandas frame for the wordEmbedding
# print(sentenceToIndicies(cleanSentence("Brdg constr. on NY 5 both directions at North West St (Syracuse) lft lane closed from 7:00AM until 4:00PM  Closure will be in the lft la...")))

'''
setupInitialData()
randKey = random.choice(twttrData.keys())
print(twttrData[randKey])
'''
# print(glovewords.index[232-1])
# print(glovewords.index[47-1])
# print(glovewords.iloc(272))
# glovewords.index.get_loc(currword)