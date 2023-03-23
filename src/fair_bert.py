import os
import argparse
import pandas as pd
import numpy as np
import csv

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

from sklearn.metrics import classification_report,f1_score

from preprocessing import preprocess_tweet


## Parameters
# LR = 2e-5
# EPOCHS = 5
# BATCH_SIZE = 32
MODEL = "bert-base-uncased"

## Data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, treatment, propensity):
        self.encodings = encodings
        self.labels = labels
        self.treatment = treatment
        self.propensity = propensity

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# def preprocess(corpus):
#     """
#     corpus: list of text strings
#     """
#     outcorpus = []
#     for text in corpus:
#         new_text = []
#         for t in text.split(" "):
#             t = t.lower()
#             t = '' if t == 'rt' else t
#             t = '' if t.startswith('@') and len(t) > 1 else t
#             t = '' if t.startswith('http') else t
#             if t != '':
#                 new_text.append(t)
#         new_text = " ".join(new_text)
#         outcorpus.append(new_text)
#     return outcorpus

def load_data(train_data_path,test_data_path):
    """
    train/test_data_path: path to csv files with columns 'text' and 'label'

    Note: sentiment analysis data for multiple languages are available here:
          # https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment
    """
    # loading training and dev dataset
    if len(train_data_path) > 0:
        df_train = pd.read_csv(train_data_path,lineterminator='\n')
        df_train = df_train[df_train.processed == ' ']
        df_val = df_train.sample(frac=0.1,random_state=33)
        df_train = df_train.drop(df_val.index)
        
    else:
        df_train = None
        df_val = None
    if len(test_data_path) > 0:
        df_test = pd.read_csv(test_data_path,lineterminator='\n')
    else:
        df_test = None
    

    dataset_dict = {
        'train':df_train,
        'val':df_val,
        'test':df_test
    }

    for i in ['train','val','test']:
        if dataset_dict[i] is not None:
            dataset_dict[i] = {
                'text':dataset_dict[i]['text'].apply(preprocess_tweet).tolist(), 
                'labels':dataset_dict[i]['label'].astype(int).tolist(),
                'treatment':dataset_dict[i]['T'].astype(int).tolist(),
                'propensity':dataset_dict[i]['propensity'].astype(float).tolist()
                }

    return dataset_dict

## fairness by causal sensitivity analysis
def ignorance_bounds(outcome, propensity, gamma):
    """
    outcome, propensity: np array
    gamma: float
    """
    lower_bound = outcome / (propensity + gamma * (1 - propensity))
    upper_bound = outcome / (propensity + 1/gamma * (1 - propensity))
    return lower_bound, upper_bound


def softmax(z):
    exp = np.exp(z - np.max(z))
    exp = exp/np.sum(exp,axis=-1)[:,np.newaxis]
    return exp


if __name__ == '__main__':
    ## command args
    parser = argparse.ArgumentParser(description='Toxity Classification by finetuning BERT.')

    parser.add_argument('--mode',default='train_and_test',type=str, help='train, test, or train_and_test')
    parser.add_argument('--train_path',default='', type=str, help='path to train/dev data, the program will automatically split train/dev')
    parser.add_argument('--test_path',default='', type=str, help='path to test data')
    parser.add_argument('--model_path',default='', type=str, help='path to trained model')
    parser.add_argument('-l','--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('-f','--max_seq_len', default=50, type=int, help='max sequence length')
    parser.add_argument('-b','--batch_size', default=32, type=int, help='mini-batch size')
    parser.add_argument('-e','--num_epoch', default = 3, type=int, help='number of epochs to train for')
    parser.add_argument('-o','--output_dir', default = './model_outputs', type=str, help='output dir to be written')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    ## Process data
    print('Start processin data...')
    data_dict = load_data(args.train_path,args.test_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True,local_files_only=True)
    for i in ['train','val','test']:
        if data_dict[i] is not None:
            encodings = tokenizer(data_dict[i]['text'], truncation=True, max_length=args.max_seq_len, padding="max_length")
            data_dict[i] = MyDataset(encodings, data_dict[i]['labels'], data_dict[i]['treatment'], data_dict[i]['propensity'])
        else:
            data_dict[i] = None

    # Args
    training_args = TrainingArguments(
        output_dir=args.output_dir,                        # output directory
        num_train_epochs=args.num_epoch,                  # total number of training epochs
        per_device_train_batch_size=args.batch_size,       # batch size per device during training
        per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
        learning_rate=args.lr,                      # learning rate
        warmup_steps=100,                         # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                        # strength of weight decay
        logging_dir=args.output_dir+'/logs',                     # directory for storing logs
        logging_steps=100,                         # when to print log
        evaluation_strategy='steps',
        eval_steps=100,
        load_best_model_at_end=True,              # load or not best model at the end
        disable_tqdm=True
    )

    ## Training
    if 'train' in args.mode:
        assert data_dict['train'] is not None, 'training data is missing!'
        print('Start training...')
        
        num_labels = len(set(data_dict["train"]["labels"]))
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels,local_files_only=True)

        trainer = Trainer(
            model=model,                              # the instantiated Transformers model to be trained
            args=training_args,                       # training arguments, defined above
            train_dataset=data_dict['train'],         # training dataset
            eval_dataset=data_dict['val']             # evaluation dataset
        )

        trainer.train()

        trainer.save_model(f"./{args.output_dir}/best_model")

        print('Finished training.')

    ## Test
    if 'test' in args.mode:
        assert data_dict['test'] is not None, 'test data is missing!'
        print('Start inferencing...')

        if args.mode == 'test':
            assert len(args.model_path) > 0, 'trained model file is missing!'
            model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
            trainer = Trainer(
                model=model,                              # the instantiated Transformers model to be trained
                args=training_args                        # training arguments, defined above
            )
        
        test_preds_raw, test_labels , _ = trainer.predict(data_dict['test'])
        test_preds_softmax = softmax(test_preds_raw)
        test_preds = np.argmax(test_preds_raw, axis=-1)
        report = classification_report(test_labels, test_preds, digits=3)
        print('NO DEBIASING:')
        print(report)
        with open(args.output_dir+'/test_preds.txt','w+') as f:
            writer = csv.writer(f)
            for i in test_preds_softmax[:,1]:
                writer.writerow([i])
        with open(args.output_dir+'/classification_report.txt','w+') as f:
            f.write(report)

        # with fairness
        best_f1 = 0
        best_gamma = 0
        unbiased_report = None
        unbiased_preds = None
        for gamma in np.arange(1.0,2.1,0.1):
            lower_bounds, upper_bounds = ignorance_bounds(test_preds_softmax[:,1],np.array(data_dict['test'].propensity),gamma)
            unbiased_test_preds_raw = [l if abs(l-0.5)<=abs(u-0.5) else u for l,u in zip(lower_bounds,upper_bounds)]
            unbiased_test_preds = [1 if i>=0.5 else 0 for i in unbiased_test_preds_raw]
            unbiased_f1 = f1_score(test_labels, unbiased_test_preds)
            if unbiased_f1 > best_f1:
                best_f1 = unbiased_f1
                best_gamma = gamma
                unbiased_report = classification_report(test_labels, unbiased_test_preds, digits=3)
                unbiased_preds = unbiased_test_preds
        print('WITH DEBIASING:')
        print(unbiased_report)
        print('best gamma=',best_gamma)
        print('fraction of original preds and debiased preds being equal=',sum([1 if i==j else 0 for i,j in zip(test_preds,unbiased_test_preds)])/len(test_preds))
        assert unbiased_report is not None, 'search over gamma failed!'
        with open(args.output_dir+'/unbiased_test_preds.txt','w+') as f:
            writer = csv.writer(f)
            for i in unbiased_test_preds_raw:
                writer.writerow([i])
        with open(args.output_dir+'/unbiased_classification_report.txt','w+') as f:
            f.write(unbiased_report)