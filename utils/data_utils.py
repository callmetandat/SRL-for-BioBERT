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
    "snips_intent_ner_to_tsv" : snips_intent_ner_to_tsv,
    "coNLL_ner_pos_to_tsv" : coNLL_ner_pos_to_tsv,
    "snli_entailment_to_tsv" : snli_entailment_to_tsv,
    "bio_ner_to_tsv" : bio_ner_to_tsv,
    "create_fragment_detection_tsv" : create_fragment_detection_tsv,
    "msmarco_query_type_to_tsv" : msmarco_query_type_to_tsv,
    "imdb_sentiment_data_to_tsv" : imdb_sentiment_data_to_tsv,
    "qqp_query_similarity_to_tsv" : qqp_query_similarity_to_tsv,
    "msmarco_answerability_detection_to_tsv" : msmarco_answerability_detection_to_tsv,
    "query_correctness_to_tsv" : query_correctness_to_tsv,
    "clinc_out_of_scope_to_tsv" : clinc_out_of_scope_to_tsv
}

class ModelType(IntEnum):
    BERT = 1

