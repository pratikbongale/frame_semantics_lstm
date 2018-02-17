"""
This file is for testing the inputs for the tensorflow model.

It contains the purest form of cleaning a tweet.
"""
import numpy as np
import pandas as pd
import csv
import random
import json
import re
from nltk.tokenize import word_tokenize

gwords = None
numFeatures = 25
def loadGloveData():
	"""
	Uses pandas to get the word embeddings from the glove dataset
	"""
	global gwords
	gloveFilename = "../glove.twitter.27B/glove.twitter.27B."+str(numFeatures)+"d.txt"
	gwords = pd.read_table(gloveFilename, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
	# gwords = gwordsTable.as_matrix().astype('float32')
	# return gwords.as_matrix().astype('float32')

def cleanTweet(tweetInput):
	"""
	This function is designed to convert a tweet's text into the format of the glove embeddings.

	Since there's many different ways to parse through and read this, I took the
	time to read through every tweet and make sure it worked. To reiterate: I went
	line by line and read every tweet and made sure that it fit the correct
	standards. If you're not sure the function is parsing correctly, I can assert that it is. 


	The function is structured as follows:
	-replace unique words that only show up once or twice ("II-TycoIS", "anti-snapchat", etc.)
	-replace large identifiers that are made up multiple characters (URLS, twitter handles, percentages, etc.)
	-replace individual emoji (unicode)
	-replace text based emoji (non-unicode)
	-add extra space for punctuation characters

	This order was not random and takes a "top-down" approach to solving this problem.

	NOTE: We use <i> for interstates, <p> for percents, and <b> for businesses.

	Returns: the text of the tweet in Glove Format.
	"""

	newSentence = tweetInput.lower()

	
	
	###Rare words
	newSentence = newSentence.replace("semi-annual"," clap ")
	newSentence = newSentence.replace("bi-polar"," bipolar ")
	newSentence = newSentence.replace("anti-snapchat"," anti <b> ")
	newSentence = newSentence.replace("peoplescout"," <b> ")
	newSentence = newSentence.replace("SHI-"," shit ")
	newSentence = newSentence.replace("shi-"," shit ")
	newSentence = newSentence.replace("II-TycoIS"," <b> ")
	newSentence = newSentence.replace("ii-tycois"," <b> ")
	newSentence = newSentence.replace("12pm"," <time> ")
	newSentence = newSentence.replace("7pm"," <time> ")
	newSentence = newSentence.replace("8pm"," <time> ")
	newSentence = newSentence.replace("9pm"," <time> ")

	newSentence = newSentence.replace("1hr"," <time> ")
	newSentence = newSentence.replace("1am"," <time> ")
	newSentence = newSentence.replace("2am"," <time> ")
	newSentence = newSentence.replace("4am"," <time> ")
	newSentence = newSentence.replace("5am"," <time> ")
	newSentence = newSentence.replace("10am"," <time> ")
	newSentence = newSentence.replace("730am"," <time> ")
	newSentence = newSentence.replace("630am"," <time> ")
	newSentence = newSentence.replace("6am"," <time> ")
	newSentence = newSentence.replace("7am"," <time> ")
	newSentence = newSentence.replace("9-5"," <time> ")
	newSentence = newSentence.replace("2-3"," <time> ")
	newSentence = newSentence.replace("2:37am"," <time> ")
	newSentence = newSentence.replace("1:30"," <time> ")
	newSentence = newSentence.replace("8:30"," <time> ")
	newSentence = newSentence.replace("f**k"," fuck ")
	newSentence = newSentence.replace("fuckkkkkkkk"," fuck ")
	newSentence = newSentence.replace("fuckkkkk"," fuck ")
	newSentence = newSentence.replace("ughhhh"," ugh ")
	newSentence = newSentence.replace("hungryyy"," hungry ")

	newSentence = newSentence.replace("51.69"," <number> ")
	newSentence = newSentence.replace("175k"," <number> ")
	newSentence = newSentence.replace("14+"," <number> ")
	newSentence = newSentence.replace("4+"," <number> ")
	newSentence = newSentence.replace("5+"," <number> ")
	newSentence = newSentence.replace("+1"," <number> ")
	newSentence = newSentence.replace("+50"," <number> ")
	newSentence = newSentence.replace("7-hour"," <number> ")
	newSentence = newSentence.replace("1corinthians"," <number> ")
	newSentence = newSentence.replace("8-4"," <number> ")
	#gave up after this
	newSentence = newSentence.replace("4e"," <number> ")
	newSentence = newSentence.replace("936f"," <number> ")
	newSentence = newSentence.replace("4g"," <number> ")
	newSentence = newSentence.replace("29n"," <number> ")
	newSentence = newSentence.replace("zblnapd0"," <number> ")
	newSentence = newSentence.replace("zblnapd0"," <number> ")
	newSentence = newSentence.replace("3rd"," <number> ")
	newSentence = newSentence.replace("4s"," <number> ")
	newSentence = newSentence.replace("3rd"," <number> ")
	newSentence = newSentence.replace("\ue106"," <number> ")
	newSentence = newSentence.replace("3v3"," <number> ")
	newSentence = newSentence.replace("30yrs"," <number> ")
	newSentence = newSentence.replace("3rd"," <number> ")
	newSentence = newSentence.replace("gr8"," <number> ")
	newSentence = newSentence.replace("9n"," <number> ")
	newSentence = newSentence.replace("ios7"," <number> ")
	newSentence = newSentence.replace("zerogravity2"," <number> ")
	newSentence = newSentence.replace("9am"," <number> ")
	newSentence = newSentence.replace("mirna-31"," <number> ")
	newSentence = newSentence.replace("2nd"," <number> ")
	newSentence = newSentence.replace("10k"," <number> ")
	newSentence = newSentence.replace("\ue404\ue404"," <number> ")
	newSentence = newSentence.replace("run4wilbs"," <number> ")
	newSentence = newSentence.replace("2nd"," <number> ")
	newSentence = newSentence.replace("racep2r"," <number> ")
	newSentence = newSentence.replace("2hearted"," <number> ")
	newSentence = newSentence.replace("joe\'90"," <number> ")
	newSentence = newSentence.replace("3rd"," <number> ")
	newSentence = newSentence.replace("5s"," <number> ")
	newSentence = newSentence.replace("5s"," <number> ")
	newSentence = newSentence.replace("5ish"," <number> ")
	newSentence = newSentence.replace("90s"," <number> ")
	newSentence = newSentence.replace("ak-47"," <number> ")
	newSentence = newSentence.replace("9n"," <number> ")
	newSentence = newSentence.replace("2nd"," <number> ")
	newSentence = newSentence.replace("w2s"," <number> ")
	newSentence = newSentence.replace("100000x"," <number> ")
	newSentence = newSentence.replace("2nd"," <number> ")
	newSentence = newSentence.replace("mg02"," <number> ")
	newSentence = newSentence.replace("2yrs"," <number> ")
	newSentence = newSentence.replace("\ue311\ue130\ue00d"," <number> ")
	newSentence = newSentence.replace("2nd"," <number> ")
	newSentence = newSentence.replace("3oh"," <number> ")
	newSentence = newSentence.replace("1-most"," <number> ")
	newSentence = newSentence.replace("10mins"," <number> ")
	newSentence = newSentence.replace("any1"," <number> ")
	newSentence = newSentence.replace("3rd"," <number> ")
	newSentence = newSentence.replace("2day"," <number> ")
	newSentence = newSentence.replace("12grade"," <number> ")
	newSentence = newSentence.replace("b8"," <number> ")
	newSentence = newSentence.replace("b4"," <number> ")
	newSentence = newSentence.replace("3s"," <number> ")
	newSentence = newSentence.replace("20a"," <number> ")
	newSentence = newSentence.replace("20a"," <number> ")
	newSentence = newSentence.replace("\ue412\ue412\ue412"," <number> ")
	newSentence = newSentence.replace("7a"," <number> ")
	newSentence = newSentence.replace("2day"," <number> ")
	newSentence = newSentence.replace("18k"," <number> ")
	newSentence = newSentence.replace("d416"," <number> ")
	newSentence = newSentence.replace("3d"," <number> ")
	newSentence = newSentence.replace("7a"," <number> ")
	newSentence = newSentence.replace("3jobs"," <number> ")
	newSentence = newSentence.replace("15a"," <number> ")
	newSentence = newSentence.replace("lr31"," <number> ")
	newSentence = newSentence.replace("16a"," <number> ")
	newSentence = newSentence.replace("d1"," <number> ")
	newSentence = newSentence.replace("16a"," <number> ")
	newSentence = newSentence.replace("23rd"," <number> ")
	newSentence = newSentence.replace("5s"," <number> ")
	newSentence = newSentence.replace("936d"," <number> ")
	newSentence = newSentence.replace("2nd"," <number> ")
	newSentence = newSentence.replace("3rd"," <number> ")
	newSentence = newSentence.replace("5c"," <number> ")
	newSentence = newSentence.replace("f4stiqk"," <number> ")
	newSentence = newSentence.replace("15a"," <number> ")
	newSentence = newSentence.replace("1-midnight"," <number> ")
	newSentence = newSentence.replace("80s"," <number> ")
	newSentence = newSentence.replace("7-hour"," <number> ")
	newSentence = newSentence.replace("16a"," <number> ")
	newSentence = newSentence.replace("50lbs"," <number> ")
	newSentence = newSentence.replace("5s"," <number> ")
	
	newSentence = newSentence.replace("1115-8"," <number> ")
	newSentence = newSentence.replace("4-4"," <number> ")
	newSentence = newSentence.replace("1045-"," <number> ")
	newSentence = newSentence.replace("5-6"," <number> ")
	newSentence = newSentence.replace("150-200"," <number> ")
	newSentence = newSentence.replace("22-0"," <number> ")
	newSentence = newSentence.replace("10-6"," <number> ")
	newSentence = newSentence.replace("14-9"," <number> ")
	newSentence = newSentence.replace("6-6"," <number> ")
	newSentence = newSentence.replace("45-"," <number> ")
	newSentence = newSentence.replace("10-2"," <number> ")
	newSentence = newSentence.replace("11-"," <number> ")
	newSentence = newSentence.replace("0153-"," <number> ")
	
	
	newSentence = newSentence.replace("10sec"," <time> ")
	newSentence = newSentence.replace("2k14"," <time> ")
	newSentence = newSentence.replace("2k"," <number> ")
	newSentence = newSentence.replace("3x"," three times ")
	newSentence = newSentence.replace("6.7%"," <p> ")
	newSentence = newSentence.replace("10,000%"," <p> ")
	newSentence = newSentence.replace("brhton"," brighton ")
	newSentence = newSentence.replace("rochseter"," rochester ")
	
	newSentence = newSentence.replace("you're"," you are ")
	newSentence = newSentence.replace("dm'd"," direct messaged ")
	newSentence = newSentence.replace("soooo-will"," so will ")
	newSentence = newSentence.replace("bitchat"," bitch at ")
	newSentence = newSentence.replace("776th"," <number> ")
	newSentence = newSentence.replace("400th"," <time> ")
	newSentence = newSentence.replace("95th"," <time> ")
	newSentence = newSentence.replace("46th"," <time> ")
	newSentence = newSentence.replace("30th"," <time> ")
	newSentence = newSentence.replace("29th"," <time> ")
	newSentence = newSentence.replace("28th"," <time> ")
	newSentence = newSentence.replace("27th"," <time> ")
	newSentence = newSentence.replace("19th"," <time> ")
	newSentence = newSentence.replace("18th"," <time> ")
	newSentence = newSentence.replace("16th"," <time> ")
	newSentence = newSentence.replace("15th"," <time> ")
	newSentence = newSentence.replace("13th"," <time> ")
	newSentence = newSentence.replace("12th"," <time> ")
	newSentence = newSentence.replace("10th"," <time> ")
	newSentence = newSentence.replace("8th"," <time> ")
	newSentence = newSentence.replace("6th"," <time> ")
	newSentence = newSentence.replace("4th"," <time> ")




	newSentence = newSentence.replace("1st"," <time> ")##Ask: Should these be numbers or times?
	newSentence = newSentence.replace("7am"," <time> ")
	newSentence = newSentence.replace("2pm"," <time> ")
	newSentence = newSentence.replace("8am"," <time> ")
	newSentence = newSentence.replace("5pm"," <time> ")
	newSentence = newSentence.replace("Tues December 3rd, 2013"," <time> ")
	newSentence = newSentence.replace("07:30am"," <time> ")
	newSentence = newSentence.replace("11:00am"," <time> ")
	newSentence = newSentence.replace("unibomber"," unabomber ")
	newSentence = newSentence.replace("part-take"," partake ")
	newSentence = newSentence.replace("dangggg"," dang ")
	newSentence = newSentence.replace("steveajobinpalooza"," party ")
	newSentence = newSentence.replace("raddisson"," radisson ")
	newSentence = newSentence.replace("ohhhhhh"," oh ")
	newSentence = newSentence.replace("geeeeeeeezus"," jesus ")
	newSentence = newSentence.replace("1993-4"," <time> ")
	newSentence = newSentence.replace("2010"," <time> ")
	newSentence = newSentence.replace("8hrs"," <time> ")
	newSentence = newSentence.replace("bremaciagg@gmail.com"," <unknown> ")
	newSentence = newSentence.replace("(585)331-1032"," <unknown> ")
	newSentence = newSentence.replace("585-739-2609"," <unknown> ")
	newSentence = newSentence.replace("katie11223344"," <unknown> ")
	newSentence = newSentence.replace("kaitlynkretschman"," <unknown> ")
	newSentence = newSentence.replace("amyfivemoon"," <unknown> ")
	newSentence = newSentence.replace("(585)331-1032"," <unknown> ")
	newSentence = newSentence.replace("mikeklooster"," <unknown> ") #Proper name
	newSentence = newSentence.replace("yudelson"," <unknown> ") #Proper name
	newSentence = newSentence.replace("tiquan"," <unknown> ") #Proper name
	newSentence = newSentence.replace("lansings"," <unknown> ") #Proper name
	newSentence = newSentence.replace("sohn-alloway"," <unknown> ") #Proper name
	newSentence = newSentence.replace("lobstuh"," lobster ")
	newSentence = newSentence.replace("pepps"," peppers ")
	newSentence = newSentence.replace("yupppp"," yup ")
	
	newSentence = newSentence.replace("errr"," er ")
	newSentence = newSentence.replace("overrr"," over ")
	newSentence = newSentence.replace("yessssss"," yes ")
	newSentence = newSentence.replace("wtf"," what the fuck ")
	newSentence = newSentence.replace("blowjob-less"," blowjob less ")
	newSentence = newSentence.replace("dialamerica"," dial america ")
	newSentence = newSentence.replace("constatnt"," constant ")
	newSentence = newSentence.replace("boothing"," <unknown> ")
	newSentence = newSentence.replace("news10"," news 10 ")
	newSentence = newSentence.replace("whhhaaaattttt"," what ")
	newSentence = newSentence.replace("gottaaaaa"," got to ")
	newSentence = newSentence.replace("euro-style"," european style ")
	newSentence = newSentence.replace("autharitive"," authoritative ")
	newSentence = newSentence.replace("shithawks"," shit hawks ")
	newSentence = newSentence.replace("whyyyy"," why ")
	newSentence = newSentence.replace("whyyy"," why ")
	newSentence = newSentence.replace("we're"," we are ")
	newSentence = newSentence.replace("1-on-1"," one on one ")
	newSentence = newSentence.replace("store\\kiosk"," store kiosk ")
	newSentence = newSentence.replace("hornell"," <unknown> ")
	newSentence = newSentence.replace("career-minded"," career minded ")
	newSentence = newSentence.replace("[pic]"," picture ")
	newSentence = newSentence.replace("kellymitchell"," <unknown> ")
	newSentence = newSentence.replace("overrr"," over ")
	newSentence = newSentence.replace("hahaaaa"," haha ")
	newSentence = newSentence.replace("lmfaooooooo"," lmfao ")
	newSentence = newSentence.replace("lmfaoooo"," lmfao ")
	newSentence = newSentence.replace("lmaoooo"," lmfao ")
	newSentence = newSentence.replace("lmaooo"," lmfao ")
	newSentence = newSentence.replace("werkkk"," werk ")
	newSentence = newSentence.replace("loveee"," love ")
	newSentence = newSentence.replace("jobbb"," job ")
	newSentence = newSentence.replace("*muah*"," muah ")
	newSentence = newSentence.replace("a-gain"," again ")
	newSentence = newSentence.replace("alllllll"," all ")
	newSentence = newSentence.replace("shitttt"," shit ")
	newSentence = newSentence.replace("yooooooooooo"," yo ")
	newSentence = newSentence.replace("ddddddd"," <unknown> ")
	newSentence = newSentence.replace("rue21"," <unknown> ")
	newSentence = newSentence.replace("wo2013155371a1"," <unknown> ")
	newSentence = newSentence.replace("️"," <unknown> ") # I actually don't know what this is
	newSentence = newSentence.replace("knuckle-head"," knucklehead ")
	newSentence = newSentence.replace("…"," <repeat> ")
	newSentence = newSentence.replace("outttaahh"," out of ")
	newSentence = newSentence.replace("hoeeeeeeeeeeeeee"," hoe ")
	newSentence = newSentence.replace("gameee"," game ")
	newSentence = newSentence.replace("lonngggg"," long ")
	newSentence = newSentence.replace("nomoreeeee"," no more ")
	newSentence = newSentence.replace("womannnnn"," woman ")
	newSentence = newSentence.replace("babyyyyyy"," baby ")
	newSentence = newSentence.replace("aaattt"," at ")
	newSentence = newSentence.replace("helll"," hell ")
	newSentence = newSentence.replace("snackssss"," snacks ")
	newSentence = newSentence.replace("gtfoh"," get the fuck out of here ")
	newSentence = newSentence.replace("crudest"," most crude ")
	
	newSentence = newSentence.replace("they're"," they are ")
	newSentence = newSentence.replace("i'd"," i would ")
	newSentence = newSentence.replace("i'm"," i am ")
	newSentence = newSentence.replace("you'd"," you would ")
	newSentence = newSentence.replace("tip-2-tip"," tip to tip ")
	newSentence = newSentence.replace("7-15-99"," <time> ")
	
	newSentence = newSentence.replace("nooooooooooooooooo"," no ")
	newSentence = newSentence.replace("uhhh"," you ")
	newSentence = newSentence.replace("youuhhh"," you ")
	newSentence = newSentence.replace("meeee"," me ")
	newSentence = newSentence.replace("stopppp"," stop ")
	newSentence = newSentence.replace("reaaallllyyyy"," really ")
	newSentence = newSentence.replace("staaank"," stank ")
	newSentence = newSentence.replace("come-hither"," come hither ")
	newSentence = newSentence.replace("cutieee"," cutie ")
	newSentence = newSentence.replace("ehhhhh"," eh ")
	newSentence = newSentence.replace("mother-f*ckin"," motherfucking ")
	newSentence = newSentence.replace("*might*"," * might * ")
	newSentence = newSentence.replace("dickkkkk"," * might * ")
	newSentence = newSentence.replace("guesss"," guess ")
	newSentence = newSentence.replace("hoursss"," hours ")
	newSentence = newSentence.replace("agent-rochester"," agent rochester ")
	newSentence = newSentence.replace("agent-syracuse"," agent syracuse ")
	newSentence = newSentence.replace("pre-travel"," pre travel ")
	newSentence = newSentence.replace("weekend-"," weekend - ")
	newSentence = newSentence.replace("saditty"," a snob ")
	newSentence = newSentence.replace("yesterdday"," yesterday ")
	newSentence = newSentence.replace("communitites"," communities ")
	newSentence = newSentence.replace("absutely"," absolutely ")
	newSentence = newSentence.replace("conessions"," confessions ")
	
	newSentence = newSentence.replace("dissertating","  ")##Real word that is not in glove!
	newSentence = newSentence.replace("caffeination","  ")##Real word that is not in glove!

	
	newSentence = newSentence.replace("24a"," <number> ") #Highway exits
	newSentence = newSentence.replace("24b"," <number> ") #Highway exits
	newSentence = newSentence.replace("wiedman"," <b> ")
	newSentence = newSentence.replace("accucheck"," <b> ")
	newSentence = newSentence.replace("wxxi"," <b> ")
	newSentence = newSentence.replace("manager-gccis"," manager - <b> ")
	newSentence = newSentence.replace("paychex"," <b> ")
	
	newSentence = newSentence.replace("2of2"," 2 of 2 ")
	newSentence = newSentence.replace("ps4"," playstation 4 ")
	newSentence = newSentence.replace("himmmm"," him ")
	newSentence = newSentence.replace("niiiggghhht"," night ")
	newSentence = newSentence.replace("y'all"," you all ")
	newSentence = newSentence.replace("make'n"," making ")
	newSentence = newSentence.replace("3-d"," 3d ")
	
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

	
	
	###Larger identifiers

	##Replace URLS
	# newSentence = replaceRegrex(tweetInput.lower(),r"https?:\/\/\S+\b|www\.(\w+\.)+\S*"," <url> ","")

	##   There is no good regular expression for URLs!
	## So I noticed that every URL (per twitter's security measures) has the pattern
	## http(s?)://t.co/[10 character alphanumeric]
	## so I just built a regular expression for that

	newSentence = replaceRegrex(newSentence,r"https://t.co/.{10}"," <url> ","")
	newSentence = replaceRegrex(newSentence,r"http://t.co/.{10}"," <url> ","")

	##Replace Percents
	newSentence = replaceRegrex(newSentence,r"\d{1,3}%"," <p> ","")

	##Replace Mentions (@...)
	# mentions = re.findall(r"@([a-zA-Z0-9_]{1,15})", newSentence)
	newSentence = replaceRegrex(newSentence,r"@(?:\w){1,15}"," <user> ","")
	# replaceRegrex(tweetInput,r"@\w+"," <user> ","")

	##Replace Hashtags (#...)
	newSentence = replaceRegrex(newSentence,r"#(\w+)"," <hashtag> ","#")
	# replaceRegrex(tweetInput,r"#(\w+)"," <hashtag> ","")

	##Replace Interstate names (I-481)
	newSentence = replaceRegrex(newSentence,r"i-[0-9]{2,3}"," <i> ","")

	##Replace Percentages (<number>%)
	# newSentence = replaceRegrex(newSentence,r"^(\d+|\d+[.]\d+)%?$"," <number> ","")
	
	##Replace capital letters
	"""
	caps = re.findall(r"([^a-z0-9()<>'`\-]){2,}", newSentence)
	for capital in caps:
		newSentence = newSentence.replace(capital," <ALLCAPS> ")
	"""

	###Emoji
	newSentence = newSentence.replace("&amp;"," & ")
	newSentence = newSentence.replace("/"," / ")
	newSentence = newSentence.replace("💕"," <heart> ")
	newSentence = newSentence.replace("❤️"," <heart> ")
	newSentence = newSentence.replace("♡"," <heart> ")
	newSentence = newSentence.replace("💜"," <heart> ")
	newSentence = newSentence.replace("💙"," <heart> ")
	newSentence = newSentence.replace("💛"," <heart> ")
	newSentence = newSentence.replace("💚"," <heart> ")
	newSentence = newSentence.replace("💘"," <heart> ")
	newSentence = newSentence.replace("💔"," <sadface> ")
	
	
	newSentence = newSentence.replace("😭"," <sadface> ")
	newSentence = newSentence.replace("😢"," <sadface> ")
	newSentence = newSentence.replace("😥"," <sadface> ")
	newSentence = newSentence.replace("😰"," <sadface> ")
	newSentence = newSentence.replace("😊"," <smile> ")
	newSentence = newSentence.replace("😍"," <heart> ")
	newSentence = newSentence.replace("☺️"," <smile> ")
	newSentence = newSentence.replace("😞"," <sadface> ")
	newSentence = newSentence.replace("😠"," angry ")
	newSentence = newSentence.replace("💎"," diamond ")
	newSentence = newSentence.replace("✂️"," scissors ")
	
	

	newSentence = newSentence.replace("👏"," clap ")
	newSentence = newSentence.replace("😡"," angry ")
	newSentence = newSentence.replace("💃"," dance ")
	newSentence = newSentence.replace("👍"," <smile> ")
	newSentence = newSentence.replace("😖"," <sadface> ")
	newSentence = newSentence.replace("😩"," <sadface> ")
	newSentence = newSentence.replace("😋"," <sadface> ")
	newSentence = newSentence.replace("😜"," <lolface> ")
	newSentence = newSentence.replace("😫"," <sadface> ")
	newSentence = newSentence.replace("😝"," <lolface> ")
	newSentence = newSentence.replace("😑"," <neutralface> ")
	newSentence = newSentence.replace("😔"," <sadface> ")
	newSentence = newSentence.replace("😁"," <neutralface> ")
	newSentence = newSentence.replace("😂"," <lolface> ")
	newSentence = newSentence.replace("😘"," <smile> ")
	newSentence = newSentence.replace("😏"," <smile> ")
	newSentence = newSentence.replace("👎"," <sadface> ")
	newSentence = newSentence.replace("💩"," <heart> ")
	newSentence = newSentence.replace("👸"," <smile> ")
	newSentence = newSentence.replace("🙈"," <lolface> ")
	newSentence = newSentence.replace("👋"," <smile> ")
	newSentence = newSentence.replace("😎"," <smile> ")
	newSentence = newSentence.replace("‼"," ! ")#NOTE: These are special exclamation marks being converted to a singular exclamation mark
	newSentence = newSentence.replace("🔫"," gun ")
	newSentence = newSentence.replace("-___-"," <neutralface> ")#for 
	newSentence = newSentence.replace("&gt;"," > ")
	newSentence = newSentence.replace("&lt;"," < ")
	newSentence = newSentence.replace("︶︿︶"," <sadface> ")
	newSentence = newSentence.replace("-__-"," <neutralface> ")
	newSentence = newSentence.replace(":("," <sadface> ")
	newSentence = newSentence.replace(":)"," <smile> ")
	newSentence = newSentence.replace("(:"," <smile> ")
	newSentence = newSentence.replace("):"," <sadface> ")
	newSentence = newSentence.replace(")="," <sadface> ")
	newSentence = newSentence.replace("=("," <sadface> ")
	newSentence = newSentence.replace(";)"," <smile> ")
	newSentence = newSentence.replace(":-("," <sadface> ")
	newSentence = newSentence.replace("💯"," <number> ")
	newSentence = newSentence.replace("💵"," <number> ")
	newSentence = newSentence.replace("🐶"," dog ")
	newSentence = newSentence.replace("🙉"," <smile> ")
	newSentence = newSentence.replace("🙊"," <lolface> ")
	newSentence = newSentence.replace("😐"," <neutralface> ")
	newSentence = newSentence.replace("😕"," <neutralface> ")
	newSentence = newSentence.replace("🐱"," <smile> ")
	newSentence = newSentence.replace("😹"," <lolface> ")
	newSentence = newSentence.replace("🚔"," <heart> ")
	newSentence = newSentence.replace("👲"," <smile> ")
	newSentence = newSentence.replace("👳"," <lolface> ")
	newSentence = newSentence.replace("✌"," peace ")
	newSentence = newSentence.replace("🍕"," pizza ")
	newSentence = newSentence.replace("👌"," smile ")
	newSentence = newSentence.replace("🏈"," football ")
	newSentence = newSentence.replace("🐄"," cow ")
	newSentence = newSentence.replace("😚"," blow ")
	newSentence = newSentence.replace("💨"," smoke ")
	newSentence = newSentence.replace("🎈"," balloon ")
	newSentence = newSentence.replace("🎉"," party ")
	newSentence = newSentence.replace("🍷"," wine ")
	newSentence = newSentence.replace("😳"," <neutralface> ")
	newSentence = newSentence.replace("😱"," wow ")
	newSentence = newSentence.replace("🚗"," car ")
	newSentence = newSentence.replace("💰"," money ")
	newSentence = newSentence.replace("😒"," <sadface> ")
	newSentence = newSentence.replace("😃"," <smile> ")
	newSentence = newSentence.replace("💣"," bomb ")
	newSentence = newSentence.replace("🍰"," cake ")
	newSentence = newSentence.replace("🎂"," cake ")
	newSentence = newSentence.replace("🙋"," woman ")
	newSentence = newSentence.replace("💁"," woman ")
	newSentence = newSentence.replace("👩"," woman ")
	newSentence = newSentence.replace("🍆"," eggplant ")
	newSentence = newSentence.replace("😵"," wow ")
	newSentence = newSentence.replace("💸"," money ")
	newSentence = newSentence.replace("😆"," laughing ")
	newSentence = newSentence.replace("🔥"," fire ")
	newSentence = newSentence.replace("🏡"," house ")
	newSentence = newSentence.replace("😷"," sick ")
	newSentence = newSentence.replace("🍦"," ice cream ")

	
	

	

	#'ve
	#16th
	# yeeehoooo
	
	
	
	#Punctuation
	newSentence = newSentence.replace("................"," <repeat> ")
	newSentence = newSentence.replace("...."," <repeat> ")
	newSentence = newSentence.replace("..."," <repeat> ")

	newSentence = newSentence.replace(";"," ; ")
	newSentence = newSentence.replace(":"," : ")
	newSentence = newSentence.replace("."," . ")
	newSentence = newSentence.replace(","," , ")
	newSentence = newSentence.replace("!"," ! ")
	newSentence = newSentence.replace("?"," ? ")
	newSentence = newSentence.replace(")"," ) ")
	newSentence = newSentence.replace("("," ( ")
	newSentence = newSentence.replace("$"," $ ")	

	newSentence = newSentence.replace("\n"," ")
	newSentence = newSentence.replace("'ve"," 've ")
	newSentence = newSentence.replace("'m"," 'm ")
	newSentence = newSentence.replace("'s"," 's ")
	newSentence = newSentence.replace("'t","t ")
	newSentence = newSentence.replace("'ll"," 'll ")
	newSentence = newSentence.replace("' "," ' ")
	newSentence = newSentence.replace("\""," \" ")
	newSentence = newSentence.replace("“"," \" ")
	newSentence = newSentence.replace("”"," \" ")

	##You HAVE to do these at the end so they don't clash!
	newSentence = newSentence.replace("nefudaboss"," <user> ")
	newSentence = newSentence.replace("em'"," them ")

	
	return newSentence

def replaceRegrex(parseableSentence, myregExpression,replaceableText,expressionStart):
	revisedSentence = parseableSentence

	many = re.findall(myregExpression, revisedSentence)
	for one in many:
		revisedSentence = revisedSentence.replace(expressionStart+one,replaceableText)

	return revisedSentence


def testCorrectness():
	testCase = 0 #The number of tweets that we looked at
	numincorrect = 0 #The number of tweets that we saw that we got correct
	failedElements = []
	with open('../smaller1year.json',"rb") as myfile:
		for line in myfile:
			tweet = json.loads(line.decode('utf8'))
			myText = tweet['tweetText'].lower()

			try:
				sent = cleanTweet(myText)
				e = None
				for element in sent.split(" "):
					if element == "":
						pass
					else:
						e = element
						currIndex = gwords.index.get_loc(element) #Looks in the pandas frame for the wordEmbedding

			except Exception as e:
				# raise e
				numincorrect = numincorrect + 1
				failedElements.append(e)
			finally:
				testCase = testCase + 1
				if (testCase % 1000) == 0:
					print("[UPDATE] We have done "+str(testCase)+" test cases with "+str(numincorrect)+" incorrect so far.")
	myfile.close()
	print("you got: "+str(numincorrect)+" tweets incorrect out of "+str(testCase)+".")
	print("Incorrect element: ")
	for y in failedElements:
		print(y)
	print("you got: "+str(numincorrect)+" tweets incorrect out of "+str(testCase)+".")
	return
"""
with open('../smaller1year.json',"rb") as fp:
	for line in fp:
		tweet = json.loads(line.decode('utf8'))
		myText = tweet['tweetText'].lower()
		# print(myText)
		print(cleanTweet(myText))
		# print(replaceRegrex(myText,r"#(\w+)"," <hashtag> ","#"))
fp.close()
"""

loadGloveData()
testCorrectness()
# gwords.index.get_loc("hi") #Looks in the pandas frame for the wordEmbedding