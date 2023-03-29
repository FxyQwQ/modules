from __future__ import absolute_import, division, print_function
import sys
# sys.path.append('../../../')
# sys.path.append('../../../python_parser')
import argparse
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup, RobertaModel,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration)
from tqdm import tqdm, trange
from model import Model

def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

LOG_FILE = '/data/yangchen/deduplication-extension/log.txt'
put_file_content(LOG_FILE, 'logging' + '\n')


# 修改
MODEL_CLASSES = {
    'codebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(js, tokenizer, args, idx):
    # source
    # code=' '.join(js['func'].split())
    code = js.split('<CODESPLIT>')[0]
    target = js.split('CODESPLIT>')[1].strip()

    # input_ids = tokenizer(code, return_tensors="pt").input_ids
    # # generated_ids = model.generate(input_ids, max_length=20)
    # source_tokens = ''
    # source_ids = input_ids
    # print(len(input_ids[0]))

    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    # print(code_tokens)
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    # print(source_tokens)
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    # print(len(source_ids))
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    # print(tokenizer.pad_token_id)
    return InputFeatures(source_tokens, source_ids, idx, int(target))


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        idx = 0
        with open(file_path) as f:
            for line in f:
                idx = idx + 1
                js = line
                try:
                    self.examples.append(convert_examples_to_features(js, tokenizer, args, idx))
                except:
                    continue
                    print('?')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    print('training...')
    # build dataloader
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    tr_loss = 0.0
    best_precision = 0
    model.zero_grad()
    for idx in range(args.epochs):
        put_file_content(LOG_FILE, "epoch :  " + str(idx) + "\n")
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            # if (step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        # global_step += 1
        # if global_step % args.save_steps == 0:
        # if global_step % 300 == 0:
        results = evaluate(args, model, tokenizer, eval_when_training=True)
        # Save model checkpoint
        if results['eval_precision'] > best_precision:
            best_precision = results['eval_precision']
            print("  " + "*" * 20)
            print("  Best acc:%s", round(best_precision, 4))
            print("  " + "*" * 20)
            put_file_content(LOG_FILE, "  " + "*" * 20 + '\n')
            put_file_content(LOG_FILE, "  Best acc: " + str(round(best_precision, 4)) + '\n')
            put_file_content(LOG_FILE, "  " + "*" * 20 + '\n')

            checkpoint_prefix = 'checkpoint-best-precision'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir,
                                        '{}'.format('model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            print("Saving model checkpoint to %s", output_dir)
        put_file_content(LOG_FILE, "\n")


def evaluate(args, model, tokenizer, eval_when_training=False):
    print('evaluating...')
    # build dataloader

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader):
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
        # if nb_eval_steps > 300:
        #     break
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = []
    for logit in logits:
        y_preds.append(np.argmax(logit))

    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1)
    }

    for key in sorted(result.keys()):
        print("  %s = %s", key, str(round(result[key], 4)))
        put_file_content(LOG_FILE, key + " = " + str(round(result[key], 4)) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # CUDA_VISIBLE_DEVICES=1,2,3 python train.py --model_name=codet5 --do_train=1 --do_retrain=0 --attack_name=our
    # parser.add_argument("--model_name", default=None, type=str, required=True,
    #                     help="model name.")
    parser.add_argument("--do_train", default=0, type=int)
    # parser.add_argument("--do_retrain", default=0, type=int)
    # parser.add_argument("--attack_name", default="", type=str,
    #                     help="attack name")
    parser.add_argument("--eval_data_file", default="/data/yangchen/deduplication-extension/datasets/gcc430-all-fixed.txt", type=str,
                        help="eval data file")
    parser.add_argument("--train_data_file", default="/data/yangchen/deduplication-extension/datasets/gcc430-all-fixed.txt", type=str,
                        help="eval data file")
    args = parser.parse_args()

    args.output_dir = './saved_models/'
    # 修改
    args.number_labels = 1
   
    args.block_size = 128
    args.seed = 123456
    args.evaluate_during_training = True
    # 修改
    args.language_type = 'c'
    args.train_batch_size = 4
    args.eval_batch_size = 1
    args.max_grad_norm = 1.0
    args.warmup_steps = 0
    args.max_steps = -1
    args.adam_epsilon = 1e-8
    args.weight_decay = 0.0
    args.gradient_accumulation_steps = 1

    args.model_type = 'codet5'
    args.config_name = 'Salesforce/codet5-base-multi-sum'
    args.model_name_or_path = 'Salesforce/codet5-base-multi-sum'
    args.tokenizer_name = 'Salesforce/codet5-base-multi-sum'
    args.epochs = 50
    args.learning_rate = 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    model = Model(model, config, tokenizer, args)

    # checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    # model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    train_dataset = TextDataset(tokenizer, args, args.train_data_file)

    # Training

    print('do train')
    train(args, train_dataset, model, tokenizer)

    print('do eval')
    checkpoint_prefix = 'checkpoint-best-precision/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)
    result = evaluate(args, model, tokenizer)
    for key in sorted(result.keys()):
        print("  %s = %s", key, str(round(result[key], 4)))
        put_file_content(LOG_FILE, key + " = " + str(round(result[key], 4)) + "\n")


if __name__ == "__main__":
    main()
