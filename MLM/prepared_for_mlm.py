import os
import re
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
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
        #predicate_token = detect_base_verbs(text, predicate)
        
        for i in range(len(words)):
            lemma, pos = analyze_word(words[i])
            if lemma == predicate:
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
    #read tsv file
    #read all files in dataDir
    files = []
    for path in os.listdir(dataDir):
        if os.path.isfile(os.path.join(dataDir, path)):
            files.append(path)
    
    write_file = open(os.path.join(wriDir, 'ner_coNLL_train.tsv'), 'w')
    for file in files:
        df = pd.read_table(os.path.join(dataDir, file), header=None)
        train, dev, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
        
        # write train to tsv file
    
        train.to_csv(write_file, sep='\t', index=False, header=False)
        
        dev.to_csv(os.path.join(wriDir, 'dev_{}.tsv'.format(file.split('.')[0])), sep='\t', index=False, header=False)
        
        test.to_csv(os.path.join(wriDir, 'test_{}.tsv'.format(file.split('.')[0])), sep='\t', index=False, header=False)
        # testW = open(os.path.join(wriDir, 'ner_coNLL_test_{}.tsv'.format(file.split('.')[0])), 'w')
        # testW.write("{}\t{}\t{}\n".format(test[0], test[1], test[2]))
        
    print(train.head())
    

convert_csv_to_tsv('../MLM/interim/', 'test_ner_output')
data_split('test_ner_output', 'coNLL_tsv')