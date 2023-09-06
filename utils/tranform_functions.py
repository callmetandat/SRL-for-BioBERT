import joblib
import argparse
import os
import re
import json
import random
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from statistics import median
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertModel

SEED = 42

def bio_ner_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):
    """
    This function transforms the BIO style data and transforms into the tsv format required
    for NER. Following transformed files are written at wrtDir,

    - NER transformed tsv file.
    - NER label map joblib file.

    For using this transform function, set ``transform_func`` : **bio_ner_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary requiring the following parameters as key-value
            
            - ``save_prefix`` (defaults to 'bio_ner') : save file name prefix.
            - ``col_sep`` : (defaults to " ") : separator for columns
            - ``tag_col`` (defaults to 1) : column number where label NER tag is present for each row. Counting starts from 0.
            - ``sen_sep`` (defaults to " ") : end of sentence separator. 
    
    """

    transParamDict.setdefault("save_prefix", "bio_ner")
    transParamDict.setdefault("tag_col", 1)
    transParamDict.setdefault("col_sep", " ")
    transParamDict.setdefault("sen_sep", "\n")

    f = open(os.path.join(dataDir,readFile))

    nerW = open(os.path.join(wrtDir, '{}_{}.tsv'.format(transParamDict["save_prefix"], 
                                                        readFile.split('.')[0])), 'w')
    labelMapNer = {}
    sentence = []
    senLens = []
    labelNer = []
    uid = 0
    print("Making data from file {} ...".format(readFile))
    for i, line in enumerate(f):
        if i%5000 == 0:
            print("Processing {} rows...".format(i))

        line = line.strip(' ') #don't use strip empty as it also removes \n
        wordSplit = line.rstrip('\n').split(transParamDict["col_sep"])
        if len(line)==0 or line[0]==transParamDict["sen_sep"]:
            if len(sentence) > 0:
                nerW.write("{}\t{}\t{}\n".format(uid, labelNer, sentence))
                senLens.append(len(sentence))
                #print("len of sentence :", len(sentence))
                sentence = []
                labelNer = []
                uid += 1
            continue
        sentence.append(wordSplit[0])
        labelNer.append(wordSplit[int(transParamDict["tag_col"])])
        if isTrainFile:
            if wordSplit[int(transParamDict["tag_col"])] not in labelMapNer:
                # ONLY TRAIN FILE SHOULD BE USED TO CREATE LABEL MAP FILE.
                labelMapNer[wordSplit[-1]] = len(labelMapNer)
    
    print("NER File Written at {}".format(wrtDir))
    #writing label map
    if labelMapNer != {} and isTrainFile:
        print("Created NER label map from train file {}".format(readFile))
        print(labelMapNer)
        labelMapNerPath = os.path.join(wrtDir, "{}_{}_label_map.joblib".format(transParamDict["save_prefix"], readFile.split('.')[0]) )
        joblib.dump(labelMapNer, labelMapNerPath)
        print("label Map NER written at {}".format(labelMapNerPath))


    f.close()
    nerW.close()

    print('Max len of sentence: ', max(senLens))
    print('Mean len of sentences: ', sum(senLens)/len(senLens))
    print('Median len of sentences: ', median(senLens))    



def coNLL_ner_pos_to_tsv(dataDir, readFile, wrtDir, transParamDict, isTrainFile=False):
    
    """
    This function transforms the data present in coNLL_data/. 
    Raw data is in BIO tagged format with the POS and NER tags separated by space.
    The transformation function converts the each raw data file into two separate tsv files,
    one for POS tagging task and another for NER task. Following transformed files are written at wrtDir

    - NER transformed tsv file.
    - NER label map joblib file.
    - POS transformed tsv file.
    - POS label map joblib file.

    For using this transform function, set ``transform_func`` : **snips_intent_ner_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.

    """

    
    f = open(os.path.join(dataDir, readFile))

    nerW = open(os.path.join(wrtDir, 'ner_{}.tsv'.format(readFile.split('.')[0])), 'w')
    posW = open(os.path.join(wrtDir, 'pos_{}.tsv'.format(readFile.split('.')[0])), 'w')

    labelMapNer = {}
    labelMapPos = {}

    sentence = []
    senLens = []
    labelNer = []
    labelPos = []
    uid = 0
    print("Making data from file {} ...".format(readFile))
    for i, line in enumerate(f):
        if i%5000 == 0:
            print("Processing {} rows...".format(i))

        line = line.strip(' ') #don't use strip empty as it also removes \n
        wordSplit = line.rstrip('\n').split(' ')
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                nerW.write("{}\t{}\t{}\n".format(uid, labelNer, sentence))
                posW.write("{}\t{}\t{}\n".format(uid, labelPos, sentence))
                senLens.append(len(sentence))
                #print("len of sentence :", len(sentence))

                sentence = []
                labelNer = []
                labelPos = []
                uid += 1
            continue
            
        sentence.append(wordSplit[0])
        labelPos.append(wordSplit[-2])
        labelNer.append(wordSplit[-1])
        if isTrainFile:
            if wordSplit[-1] not in labelMapNer:
                # ONLY TRAIN FILE SHOULD BE USED TO CREATE LABEL MAP FILE.
                labelMapNer[wordSplit[-1]] = len(labelMapNer)
            if wordSplit[-2] not in labelMapPos:
                labelMapPos[wordSplit[-2]] = len(labelMapPos)
    
    print("NER File Written at {}".format(wrtDir))
    print("POS File Written at {}".format(wrtDir))
    #writing label map
    if labelMapNer != {} and isTrainFile:
        print("Created NER label map from train file {}".format(readFile))
        print(labelMapNer)
        labelMapNerPath = os.path.join(wrtDir, "ner_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapNer, labelMapNerPath)
        print("label Map NER written at {}".format(labelMapNerPath))

    if labelMapPos != {} and isTrainFile:
        print("Created POS label map from train file {}".format(readFile))
        print(labelMapPos)
        labelMapPosPath = os.path.join(wrtDir, "pos_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapPos, labelMapPosPath)
        print("label Map POS written at {}".format(labelMapPosPath))

    f.close()
    nerW.close()
    posW.close()

    print('Max len of sentence: ', max(senLens))
    print('Mean len of sentences: ', sum(senLens)/len(senLens))
    print('Median len of sentences: ', median(senLens))


model = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2',
                                    output_hidden_states = True # Whether the model returns all hidden-states.
                                    )

def read_data(readPath):

    with open(readPath, 'r', encoding = 'utf-8') as file:
        taskData = []
        for i, line in enumerate(file):
            sample = json.loads(line)
            taskData.append(sample)
            
    return taskData

def get_embedding(dataDir, readFile, wrtDir):
    
    data = read_data(os.path.join(dataDir, readFile))
    
    vecs_wri = open(os.path.join(wrtDir, 'vecs_{}.json'.format(readFile.split('.')[0])), 'w')
    
    for i, line in enumerate(data):
        tokens_id = line['token_id']
        segments_id = line['type_id']
        u_id = line['uid']
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([tokens_id])
        segments_tensors = torch.tensor([segments_id])
        
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
           
        # `hidden_states` is a Python list.
        # Each layer in the list is a torch tensor.
        # `token_vecs` is a tensor with shape [50 x 768]
        
        token_vecs = hidden_states[-2][0]
        
        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        
        vecs_wri.write("{}\t{}\n".format(u_id, sentence_embedding))
        print("Processed {} rows...".format(i))
    
