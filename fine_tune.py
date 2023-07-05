import argparse, os, logging, torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from transformers import BertTokenizer
TF_ENABLE_ONEDNN_OPTS=0

logger = logging.getLogger(__name__)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
BERT_MODEL = 'dmis-lab/biobert-base-cased-v1.2'
def main():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True,truncation=True)
    print(tokenizer)
    # BertTokenizer(name_or_path='dmis-lab/biobert-base-cased-v1.2', vocab_size=28996, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True)
    

if __name__ == "__main__":
    main()


    