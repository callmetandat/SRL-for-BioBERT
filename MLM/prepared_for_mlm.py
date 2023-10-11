import os
import re
import numpy as np
import pandas as pd
import json
from ast import literal_eval
from sklearn.model_selection import train_test_split
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def extract_predicate_from_filename(filename):
    # Extract the predicate from the filename (assuming the filename format [predicate]_full.csv)
    base_name = os.path.basename(filename)
    predicate = os.path.splitext(base_name)[0].split('_')[0]
    return predicate

# Define a function to detect base verbs
def analyze_word(word, lowercase=True):
    tokens = nlp(word)
    lemma = tokens[0].lemma_
    if lowercase: lemma = lemma.lower()
    return lemma, tokens[0].pos_

def detect_base_verbs(sentence, predicate):
    # Process the input sentence with spaCy
    doc = nlp(sentence)

    # Iterate through the tokens in the sentence
    for i, token in enumerate(doc):
        
        # Check if the token is a verb (POS tag starts with 'V') and not a auxiliary verb (aux)
        if token.pos_.startswith('V') and token.dep_ != 'aux':
            if token.lemma_ == predicate:
                return token.text

    return None

def convert_to_ner_format(df, file):
    ner_format = []
    words_tokenized = []
     # Extract the predicate from the filename
   
    predicate = extract_predicate_from_filename(file)
    
    # convert arguments to dictionary
    df['arguments'] = df['arguments'].apply(lambda x: eval(x))
    for index, row in df.iterrows():
        
        text = str(row['text']).lower()
        arguments = row['arguments']
        
        # Tokenize the text into words while preserving selected punctuation
        words = re.split(r'([.,;\s])', text)    
        words = [word for word in words if word != ' ' and word != ''] 
    
        ner_tags = ['O'] * len(words)
        
        # Detect the predicate in the sentence
        # Predicate_token = detect_base_verbs(text, predicate)
        
        for i in range(len(words)):
            lemma, pos = analyze_word(words[i])
            if (lemma == predicate):
                words[i] = '#' + words[i]
                ner_tags[i] = 'B-V'
                break
         
        for arg_id, arg_text in arguments.items():           
            arg_text = arg_text.lower()
            # Tokenize the argument into words
            arg_words = re.split(r'([.,;\s])', arg_text)    
            arg_words = [word for word in arg_words if word != ' ' and word != '']
            
            # Iterate through the words in the sentence
            for i in range(len(words) - len(arg_words) + 1):
                if words[i:i+len(arg_words)] == arg_words:
                    # Assign a label based on the argument key
                    for j in range(len(arg_words)):
                        if j == 0:
                            ner_tags[i+j] = f'B-A{arg_id}'
                        else:
                            ner_tags[i+j] = f'I-A{arg_id}'
        ner_format.append(ner_tags)
        words_tokenized.append(words)
    return ner_format, words_tokenized

def convert_csv_to_tsv(readDir, writeDir):
    # read csv file, ignore first line
    # read all files in readDir
    files = []
    for path in os.listdir(readDir):
        if os.path.isfile(os.path.join(readDir, path)):
            files.append(path)
            
    for file in files:
        data_df = pd.read_csv(os.path.join(readDir, file), sep=',', header=0)
        ner_format = convert_to_ner_format(data_df, file)
        labelNer = ner_format[0]
        uid = data_df['id']                      
        text = ner_format[1]
        nerW = open(os.path.join(writeDir, 'ner_{}.tsv'.format(file.split('.')[0])), 'w')
        for i in range(len(uid)):
            nerW.write("{}\t{}\t{}\n".format(uid[i], labelNer[i], text[i]))
    nerW.close()
    
def data_split(dataDir, wriDir):
    #read json
    #read all files in dataDir
    files = []
    for path in os.listdir(dataDir):
        if os.path.isfile(os.path.join(dataDir, path)):
            files.append(path)

    df = pd.DataFrame()
    for file in files:
        f = open(os.path.join(dataDir, file))
        json_data = pd.read_json(f, lines=True)
        #train, dev, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        train, testt = train_test_split(json_data, test_size=0.4)
        dev, test = train_test_split(testt, test_size=0.5)
        
        df = pd.concat([df, train], ignore_index=True)
        
        print("Processing file: ", file)
        dev.to_json(os.path.join(wriDir, 'dev_{}.json'.format(file.split('.')[0])), orient='records', lines=True)
        test.to_json(os.path.join(wriDir, 'test_{}.json'.format(file.split('.')[0])), orient='records', lines=True)   
        
    df.to_json(os.path.join(wriDir, 'ner_coNLL_train.json'), orient='records', lines=True)
   
#convert_csv_to_tsv('../MLM/interim/', 'test_ner_output')
#data_split('mlm_output', 'mlm_prepared_data')

def create_data_mlm(dataDir, wriDir, tokenizer, maxSeqLen):
    '''
    Function to create data in NER/Sequence Labelling format.
    The tsv format expected by this function is 
    sample['uid] :- unique sample/sentence id
    sample['sentence'] :- list of the sentence tokens for the sentence eg. ['My', 'name', 'is', 'hello']
    sample['label] :- list of corresponding tag for token in sentence ed. ['O', 'O', 'O', 'B-A1']
    '''
    
    # Read train file
    files = []
    for path in os.listdir(dataDir):
        if os.path.isfile(os.path.join(dataDir, path)):
            files.append(path)
    
    for file in files:
        writefile = os.path.join(wriDir, 'mlm_{}.json'.format(file.split('.')[0]))
        with open(writefile, 'w') as wf:
            data = pd.read_csv(os.path.join(dataDir, file))
            print("Processing file: ", file)
            
            for idx, sample in enumerate(data['text']) :    
                uids = data['id'][idx]   

                # Process the tokenized text with spaCy
                doc = nlp(sample)

                # Get POS tags for each token
                pos_tags = [token.pos_ for token in doc]

                out = tokenizer.encode_plus(text = sample, add_special_tokens=True,
                                    truncation_strategy ='only_first',
                                    max_length = maxSeqLen, pad_to_max_length=True)
                # print(out.keys())  dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
                typeIds = None
                inputMask = None
                tokenIds = out['input_ids']
                if 'token_type_ids' in out.keys():
                    typeIds = out['token_type_ids']
                if 'attention_mask' in out.keys():
                    inputMask = out['attention_mask']
                           
                assert len(tokenIds) == maxSeqLen, "Mismatch between processed tokens and labels"
                    # feature = {'uid': uids, 'token_id': tokenIds[0], 'mask': inputMask[0]}
                feature = {'uid':str(uids), 'token_id': tokenIds, 'mask': inputMask, 'pos': pos_tags}
                    
                wf.write('{}\n'.format(json.dumps(feature)))
            print("Done file: ", file)
    
def masking_sentence(token_ids):
    for token in token_ids:
        
    return 
# Chia train test ngay từ đầu         
# 1. Hàm xử lí từng câu(duyệt qua từng token trong câu) - for để mask từng token 
# 
# 2. Hàm xử lí file train(duyệt qua từng câu trong file) => return df
#
# 3. Hàm train truyền vào df, return model
#
# data gom: id, token_id, attention_mask, pos 
# df gom: id, masked_token_id, label 

'''
text = "Who was Jim Paterson ? Jim Paterson is a doctor".lower()
inputs  =  tokenizer.encode_plus(text,  return_tensors="pt", add_special_tokens = True, truncation=True, pad_to_max_length = True,
                                         return_attention_mask = True,  max_length=64)
input_ids = inputs['input_ids']
labels  = copy.deepcopy(input_ids) #this is the part I changed
input_ids[0][7] = tokenizer.mask_token_id
labels[input_ids != tokenizer.mask_token_id] = -100 

loss, scores = model(input_ids = input_ids, attention_mask = inputs['attention_mask'] , token_type_ids=inputs['token_type_ids'] , labels=labels)
print('loss',loss)


["DET", "NOUN", "PUNCT", "ADP", "PUNCT", "NOUN", "NOUN", "ADP", "DET", "ADJ", "NOUN", "ADP", "NOUN", "NUM", "ADP", "NOUN", "NUM", "VERB", "ADJ", "NOUN", "PUNCT"]
21
A G-to-A transition at the first nucleotide of intron 2 of patient 1 abolished normal splicing.
'''
# df_train: [input_ids, attention_mask, label]
# file data train bự:
# input_ids, attention_mask, label

# from transformers import AutoTokenizer
# model_checkpoint = "dmis-lab/biobert-base-cased-v1.2"

# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# create_data_mlm('./interim/', './mlm_output/', tokenizer, 50)