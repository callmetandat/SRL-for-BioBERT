import copy
import os
import re
import numpy as np
import pandas as pd
import json
from ast import literal_eval
from sklearn.model_selection import train_test_split
import spacy

from transformers import BertTokenizer #chạy đc xíu r kìa
MAX_SEQ_LEN = 50

wwm_probability = 0.1
# Load the English language model
nlp = spacy.load("en_core_web_sm")

def get_files(dir):
    files = []
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            files.append(path)
    return files

def convert_csv_to_tsv(readDir, writeDir):
    '''
    convert_csv_to_tsv('../MLM/interim/', 'coNLL_tsv')
    '''
    def convert_to_ner_format(df, file):
        ner_format = []
        words_tokenized = []
        
        # Extract the predicate from the filename
        base_name = os.path.basename(file)
        predicate = os.path.splitext(base_name)[0].split('_')[0]
        
        # convert arguments to dictionary
        df['arguments'] = df['arguments'].apply(lambda x: eval(x))
        for index, row in df.iterrows():
            text = str(row['text']).lower()
            arguments = row['arguments']
            
            # Tokenize the text into words while preserving selected punctuation
            words = re.split(r'([.,;\s])', text)    
            words = [word for word in words if word != ' ' and word != ''] 
        
            ner_tags = ['O'] * len(words)
            
            for i in range(len(words)):
                tokens = nlp(words[i])
                
                if (tokens[0].lemma_.lower() == predicate):
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
    
    files = get_files(readDir)
            
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
 
def tokenize_csv_to_json(dataDir, wriDir, tokenizer):
    '''
    Tokenize_csv_to_json('./interim/', './mlm_output/', tokenizer)
    Function to create data in MLM format.
    Input file: csv with columns ['id', 'source ,'text', 'arguments']
    Output file: json with columns ['uid', 'token_id', 'mask', 'pos']
    
    '''
    
    # Read train file
    files = get_files(dataDir)
    
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
                                    max_length = MAX_SEQ_LEN, pad_to_max_length=True)
                
                inputMask = None
                tokenIds = out['input_ids']
                
                if 'attention_mask' in out.keys():
                    inputMask = out['attention_mask']
                           
                assert len(tokenIds) == MAX_SEQ_LEN, "Mismatch between processed tokens and labels"
                
                feature = {'uid':str(uids), 'token_id': tokenIds, 'mask': inputMask, 'pos': pos_tags}
                    
                wf.write('{}\n'.format(json.dumps(feature)))
            print("Done file: ", file)

def masking_sentence(token_ids, tokenizer):
        '''
        Function to mask random token in a sentence and return the masked sentence and the corresponding label ids
        '''
        except_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
        
        mask = np.random.binomial(1, 0.1, (len(token_ids),))   # Masking each token with 15% probability
        
        token_ids_copy = copy.deepcopy(token_ids)
        
        labels = [-100] * len(token_ids)
        for idx, token in enumerate(token_ids):
            
            if (idx in np.where(mask)[0]) and (token not in except_tokens):
                token_ids_copy[idx] = tokenizer.mask_token_id
                labels[idx] = token
            
        return token_ids_copy, labels
    
def masked_df(df, tokenizer):
    '''
    Function to mask tokens in a dataframe and return the masked dataframe.
    df has columns ['uid', 'token_id', 'mask', 'pos', 'masked_token_id', 'label_id']
    '''
    masked_sentences = []
    labels = []
    for idx, sen in enumerate(df['token_id']):
        masked_sentence, label = masking_sentence(sen, tokenizer)

        masked_sentences.append(masked_sentence)
        labels.append(label)
    
    df['masked_token_id'] = masked_sentences
    df['label_id'] = labels
    return df

def data_split(dataDir, wriDir, tokenizer):
    
    '''
    data_split('mlm_output', 'mlm_prepared_data', tokenizer)
    Function to split data into train, dev, test (60, 20, 20) and write to json files.
    '''
    files = get_files(dataDir)

    train_df = pd.DataFrame()
    for file in files:
        f = open(os.path.join(dataDir, file))
        json_data = pd.read_json(f, lines=True)
        
        train, testt = train_test_split(json_data, test_size=0.4)
        dev, test = train_test_split(testt, test_size=0.5)
        train_df = pd.concat([train_df, train], ignore_index=True)
        
        dev_df = masked_df(dev, tokenizer)
        test_df = masked_df(test, tokenizer)
        
        print("Processing file: ", file)
        dev_df.to_json(os.path.join(wriDir, 'dev_{}.json'.format(file.split('.')[0])), orient='records', lines=True)
        test_df.to_json(os.path.join(wriDir, 'test_{}.json'.format(file.split('.')[0])), orient='records', lines=True)   
    
    train_df = masked_df(train_df, tokenizer)    
    train_df.to_json(os.path.join(wriDir, 'train_mlm.json'), orient='records', lines=True)


def main():
    tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    #tokenize_csv_to_json('./interim/', './mlm_output/', tokenizer)
    data_split('mlm_output', 'mlm_prepared_data', tokenizer)
    
   
if __name__ == "__main__":
    main() 
    
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
