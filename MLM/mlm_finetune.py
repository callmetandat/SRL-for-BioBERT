from argparse import ArgumentParser
import math
from pathlib import Path
import torch
import logging
import json
import random
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
from tempfile import TemporaryDirectory
import torch.nn as nn
import torch
import sys
from scipy.special import softmax

sys.path.insert(1, '../')

# import sys
# sys.path.insert(1, '/content/SRL-for-BioBERT')
from embedding import read_data
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
# from bert_mlm_finetune import BertForMLMPreTraining 
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup, BertForMaskedLM
from utils_mlm import count_num_cpu_gpu
from prepared_for_mlm import data_split
import spacy
import pathlib
import torch.nn.functional as F
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", do_lower_case=True)

MLM_IGNORE_LABEL_IDX = -1
BATCH_SIZE = 32
EPOCHS = 10
FP16 = True
NUM_CPU = count_num_cpu_gpu()[0]

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        train_file = training_path / "train_mlm.json"
        assert train_file.is_file() 
        data_list = []
        with open(train_file) as f:
            for line in f:
                data = json.loads(line)
                data_list.append(data)
        # num_samples = metrics['num_training_examples']
        train_df = pd.DataFrame(data_list)
        
        num_samples = len(train_df)
        seq_len = 50
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            print("reduce memory")
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            self.input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            self.lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            #lm_label_ids[:] = MLM_IGNORE_LABEL_IDX
        else:
            self.input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            self.input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            self.lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=MLM_IGNORE_LABEL_IDX)
        logging.info(f"Loading training examples for epoch {epoch}")
        # with data_file.open() as f:
        #     for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
        #         line = line.strip()
        #         example = json.loads(line)
        #         features = convert_example_to_features(example, tokenizer, seq_len)
        #         input_ids[i] = features.input_ids
        #         input_masks[i] = features.input_mask
        #         lm_label_ids[i] = features.lm_label_ids
        # assert i == num_samples - 1  # Assert that the sample count metric was true
        
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        #
        self.input_ids = train_df['token_id']
        self.input_masks = train_df['attention_mask']
        self.lm_label_ids = train_df['labels']
        
       
    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(np.array(self.input_ids[item]), dtype = torch.int64),
                torch.tensor(np.array(self.input_masks[item]), dtype = torch.int64),
                torch.tensor(np.array(self.lm_label_ids[item]), dtype = torch.int64))

def get_pos_tag(text, index):
    # Process the text with spaCy
    doc = nlp(text)

    # Check if the index is within the bounds
    if index < 0 or index >= len(doc):
       return "Index out of bounds"

    # Get the POS tag for the word at the specified index
    pos_tag = doc[index].pos_
    return pos_tag

def is_POS_match(logits, input_ids, lm_label_ids):
    '''
    Function to check if the POS tag of the masked token in the logits is the same as the POS tag of the masked token in the original text.
    '''
    origin_input_id = input_ids
    
    # Create a pandas dataframe
    df = pd.DataFrame({"values": lm_label_ids})

    # Find the index of the first element not equal to -100
    masked_idx = df.loc[df["values"] != -100].index[0]
    
    for i in range(len(origin_input_id)):
        if origin_input_id[i] == 103:
            origin_input_id[i] = lm_label_ids[masked_idx]
    
    # get pos tag of origin text
    origin_text = tokenizer.decode(origin_input_id)
   
    pos_tag_origin = get_pos_tag(origin_text, masked_idx)
 
    # get pos tag of logits
    logits_tag = get_pos_tag(tokenizer.decode(logits), masked_idx)
   
    return pos_tag_origin == logits_tag    

def custom_loss(input_ids, logits, labels):
    # loss = 0.5 * (-torch.log(1 - softmax(logits))[labels]) + (1 - is_POS_match(logits=logits, input_ids=input_ids, lm_label_ids=labels))
    # return loss.mean()
    # Cross-entropy term
    
    cross_entropy_term = F.cross_entropy(logits, labels)

    # Custom matching term
    matching_term = is_POS_match(logits=logits, input_ids=input_ids, lm_label_ids=labels)

    # Combine terms
    loss = 0.5 * cross_entropy_term + (1 - matching_term)

    return loss

def pretrain_on_treatment(args):
    # assert args.pregenerated_data.is_file(), \
    #     "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    print("args.train_batch_size", args.train_batch_size)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)
 
    # total_train_examples = 0
    # for i in range(args.epochs):
    #     # The modulo takes into account the fact that we may loop over limited epochs of data
    #     total_train_examples += samples_per_epoch[i % len(samples_per_epoch)] 
     
    # num_train_optimization_steps = int(
    #     total_train_examples / args.train_batch_size / args.gradient_accumulation_steps) 
    num_train_optimization_steps = math.ceil(args.num_samples/args.train_batch_size) * args.epochs // args.gradient_accumulation_steps
    
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size() 
    
    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    if args.fp16:
        model.half()
        
    model.to(device)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
   
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f" Num examples = {args.num_samples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    loss_dict = defaultdict(list)
    for epoch in range(args.epochs):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=args.epochs, reduce_memory=args.reduce_memory)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=NUM_CPU)
        
       
        tr_loss = 0 
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                
                batch = tuple(t.to(device) for t in batch)               
                input_ids, input_mask, lm_label_ids = batch              
                outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=lm_label_ids)
                # outputs: loss, logits, hidden_states, attentions
               
                # print top 10 masked tokens
                # print(tokenizer.convert_ids_to_tokens(torch.topk(outputs.logits[0, idx, :], 10).indices))
                #print("Input id shape: ", input_ids.shape)  
                #print("Logits shape: ", outputs.logits.shape) # ([32, 85, 28996])  input (N=batch_sz, C=nb_of_class)
                # logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) 
                #print("Target shape: ", lm_label_ids.shape)  # torch.Size([32, 85]) Target:  shape (), (N)
                
                num_classes = outputs.logits[0].size(1) # Output:  shape (), (N)
                #print("Num classes ", num_classes) # 28996
                loss = custom_loss(input_ids=input_ids, logits=outputs.logits.view(-1, num_classes), labels=lm_label_ids.view(-1))
                
                #loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1
                loss_dict["epoch"].append(epoch)
                loss_dict["batch_id"].append(step)
                loss_dict["mlm_loss"].append(loss.item())
        # Save a trained model
        if epoch < args.epochs and (n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <= 1):
            logging.info("** ** * Saving fine-tuned model ** ** * ")
            epoch_output_dir = args.output_dir / f"epoch_{epoch}"
            epoch_output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)

    # Save a trained model
    if n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <=1:
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        df = pd.DataFrame.from_dict(loss_dict)
        df.to_csv(args.output_dir/"losses.csv")
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=False)
    parser.add_argument("--output_dir", type=Path, required=False)
    parser.add_argument("--bert_model", type=str, required=False, default='dmis-lab/biobert-base-cased-v1.2',
                        help="Bert pre-trained model")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--num_samples", type=int, required=False, default=1000000)
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=BATCH_SIZE,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=222,
                        help="random seed for initialization")
    parser.add_argument("--corpus_type", type=str, required=False, default="")
    args = parser.parse_args()

    args.output_dir = Path('mlm_finetune_output') / "model"
    args.pregenerated_data = pathlib.Path('mlm_prepared_data')
    
    # data_split('mlm_output', 'mlm_prepared_data', tokenizer)[0]
    pretrain_on_treatment(args)
   

if __name__ == '__main__':
    main()