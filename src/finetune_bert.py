import os
import argparse
import pandas as pd
import numpy as np
import csv

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

from sklearn.metrics import classification_report


## Parameters
# LR = 2e-5
# EPOCHS = 5
# BATCH_SIZE = 32
MODEL = "bert-base-uncased"

## Data
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocess(corpus):
    """
    corpus: list of text strings
    """
    outcorpus = []
    for text in corpus:
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        new_text = " ".join(new_text)
        outcorpus.append(new_text)
    return outcorpus

def load_data(train_data_path,test_data_path):
    """
    train/test_data_path: path to csv files with columns 'text' and 'label'

    Note: sentiment analysis data for multiple languages are available here:
          # https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment
    """
    # loading training and dev dataset
    df_train = pd.read_csv(train_data_path,lineterminator='\n')
    df_test = pd.read_csv(test_data_path,lineterminator='\n')
    df_val = df_train.sample(frac=0.1,random_state=33)
    df_train = df_train.drop(df_val.index)

    dataset_dict = {
        'train':df_train,
        'val':df_val,
        'test':df_test
    }

    for i in ['train','val','test']:
        dataset_dict[i] = {
            'text':preprocess(dataset_dict[i]['text'].tolist()), 
            'labels':dataset_dict[i]['label'].astype(int).tolist()
            }

    return dataset_dict



if __name__ == '__main__':
    ## command args
    parser = argparse.ArgumentParser(description='Toxity Classification by finetuning BERT.')

    parser.add_argument('--train_path', type=str, help='path to train/dev data, the program will automatically split train/dev')
    parser.add_argument('--test_path', type=str, help='path to test data')
    parser.add_argument('-l','--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('-f','--max_seq_len', default=50, type=int, help='max sequence length')
    parser.add_argument('-b','--batch_size', default=32, type=int, help='mini-batch size')
    parser.add_argument('-e','--num_epoch', default = 3, type=int, help='number of epochs to train for')
    parser.add_argument('-o','--output_dir', default = './model_outputs', type=str, help='output dir to be written')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    ## process data
    print('Start processin data...')
    dataset_dict = load_data(args.train_path,args.test_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True,local_files_only=True)
    train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, max_length=args.max_seq_len, padding="max_length")
    val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, max_length=args.max_seq_len, padding="max_length")
    test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, max_length=args.max_seq_len, padding="max_length")

    train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
    val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])
    test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])

    ## Training
    print('Start training...')
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

    num_labels = len(set(dataset_dict["train"]["labels"]))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels,local_files_only=True)

    trainer = Trainer(
        model=model,                              # the instantiated Transformers model to be trained
        args=training_args,                       # training arguments, defined above
        train_dataset=train_dataset,              # training dataset
        eval_dataset=val_dataset                  # evaluation dataset
    )

    trainer.train()

    trainer.save_model(f"./{args.output_dir}/best_model")

    print('Finished training.')

    ## Test
    test_preds_raw, test_labels , _ = trainer.predict(test_dataset)
    test_preds = np.argmax(test_preds_raw, axis=-1)
    report = classification_report(test_labels, test_preds, digits=3)
    print(report)

    with open(args.output_dir+'/test_preds.txt','w+') as f:
        writer = csv.writer(f)
        for i in test_preds:
            writer.writerow([i])

    with open(args.output_dir+'/classification_report.txt','w+') as f:
        f.write(report)