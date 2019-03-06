import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time
import os
import csv

def main(argv):
	path = "Release"

	# Write bow file
	numPairs = 0
	with open(path + "/" + "RawData/[filename]", 'r') as f:
		lines = f.readlines()

	with  open(path+"/"+"myProcess/[new filename]", "w") as w:
		for line in lines:
			s5 = line.split("|||")
			source = s5[0].split("\t")[2]
			l = len(s5)
			for i in range(l-1):
				s = s5[i+1].split("\t")
				numRatings = int(s[2])
				allGood = True
				for i in range(numRatings):
					if (int(s[i*2+3]) != 6):
						allGood = False
				if allGood == True:
					numPairs +=1
				w.write(source + "\t" + s[0] +"\n")

	print("New number of training pairs " + str(numPairs))

if __name__ == "__main__":
  main(sys.argv[1:])

