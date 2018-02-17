# Files in directory
  - README.md - this file
  - main.py - Contains the encoder/decoder model and f1 score calculation
  - loaddata.py - Contains the code for loading embeddings and tweets, and generates tweets into an appropriate form for the model
  - textlookup.py - Script file that

# How to install code (what programming lnguages/versopms/where to find/ what libraries)

This project uses python 3 and was built with following libraries:

  - numpy 1.13.3
  - pandas 0.21.0
  - tensorflow 1.4.0

These libraries can be installed with the commands:
```sh
$ pip3 install numpy
$ pip3 install tensorflow
$ pip3 install pandas
```

# Where to find data
The data can be found in the following files: [new1year.json](https://drive.google.com/file/d/0B3DvP_3g40x4LTgzdnJuWHFockE/view) which contains all of the tweets and [combined_annotations.json](https://drive.google.com/file/d/0B3DvP_3g40x4QzJIX2g0Z1N6WVk/view) which contains the classes for all of the tweets. We relate the data in these two files by the ID of the tweet.


# How to run code
Given the initial dataset with the files 'new1year.json' and 'combined_annotations.json', you need to separate it into the singular json file that the program can understand. There is a script in the directory which does this. Save the files in this directory and run the following commands:

```sh
$ cd take2
$ python3 cleaner.py smallersplit
$ python3 cleaner.py unknownIndicies
```


To run the model, run the following commands:

```sh
$ cd take2
$ python main.py trainterm
```
