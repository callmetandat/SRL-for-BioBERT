from enum import IntEnum
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import XLNetConfig, XLNetModel, XLNetTokenizer
from transformers import ElectraConfig, ElectraModel, ElectraTokenizer
from utils.tranform_functions import *

from transformers import BertTokenizer
bioBertTokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2', do_lower_case=True,truncation=True)

NLP_MODELS = {
    "bert": (BertConfig, BertModel, bioBertTokenizer, 'bert-base-uncased'),
}

TRANSFORM_FUNCS = {
    "coNLL_ner_pos_to_tsv" : coNLL_ner_pos_to_tsv,
    "bio_ner_to_tsv" : bio_ner_to_tsv,
}

class ModelType(IntEnum):
    BERT = 1

