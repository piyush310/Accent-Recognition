# The file contains the code to recognise accent 
# Here for simplicity only two accent have been used Indian and Russian
# It reuires 3 folders: 2 for training and 1 for testing




import numpy as np
from collections import Counter
from pylab import *
import warnings
from scipy.io import wavfile as wv
from scipy.cluster.vq import kmeans,vq
import python_speech_features as sf
from pylab import *
import pandas as pd
warnings.filterwarnings("ignore")

#file1 and file3 are the training files  
#file2 is the testing file
# All files must be in wav format

file1 = "./hindi_wav/hindi"		# This folder for consist of 10 files 
file2 = "./russian_wav/russian"	# This folder has 10 files
file3 = "./test_wav/test"		# This folder has 12 files


#####################################################
#				Initialization						# 
#####################################################
lifter = 0
numcep = 25
v=2
code=[]

sb=[]

#####################################################
#				Feature Selection		 			#
#####################################################

def feat(wav,c=False,code=[],lifter=0,numcep=25,v=2):
	fs,s=wv.read(wav)
	mf=sf.mfcc(s,samplerate=fs,numcep=numcep,ceplifter=lifter)
	
	norm_feat=[]
	for i,feat in enumerate(mf):
		norm_feat.append((feat-np.mean(feat))/np.std(feat))
		
	if c==True:
		codebook, distortion = kmeans(norm_feat, v)
	else:
		codebook = code
	codewords, dist = vq(norm_feat, codebook)
	sb.append(codewords)
	histo = np.array(list(Counter(codewords).values()))#/len(mf)
	# print(wav,"\t",histo)
	return histo,codebook,sb




#######################################################################
#					Feature Matrix for Indian Accent 				  #
#######################################################################
hind= np.zeros(11)
hindi_count=0 
a = np.zeros((11,2))
for i in range(1,11):
	file=file1+ str(i) + ".wav"
	#print file
	a[i],code,sbp = feat(file,True,v=v)
	

########################################################################
#					Feature Matrix for Russian Accent 			   	   #
########################################################################
hind= np.zeros(11)

russ = np.zeros(11)
russia_count=0
ru = np.zeros((11,2))
for i in range(1,11):
	file=file2+ str(i) + ".wav"
	#print file
	ru[i],code,sbp = feat(file,True,v=v)

#########################################################################
#					Feature Matrix for Test 							#
#########################################################################
b = np.zeros((13,2))
for j in range(1,13):
	file =file3+ str(j) + ".wav"
	b[j],code,sbp = feat(file,True,v=v)


#########################################################################
#	   Checks for each test file and prints which accent it is			#
#########################################################################
for i in range(1,13):
	hindi_count=0
	russia_count = 0 
	for j in range(1,11):
		hindi_count += np.linalg.norm(a[j]-b[i])
		russia_count += np.linalg.norm(ru[j]-b[i])
	
	if russia_count < hindi_count:
		print "test"+str(i)	+ " Russian"
	else:
		print "test"+str(i)	+ " Indian"

