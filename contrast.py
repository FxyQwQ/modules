# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
import sys

class CodeT5RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, features, **kwargs):
        # x = features.reshape(-1, features.size(-1))
        x = self.dense1(features)
        x = F.tanh(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.bugNumber = 4
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = CodeT5RobertaClassificationHead(config)
        self.args = args
        self.query = 0

        self.temperature = 0.5
        self.scale_by_temperature = True

        self.out_proj = nn.Linear(config.hidden_size, self.bugNumber)

    def forward(self, input_ids=None, labels=None):

        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                  hidden_states.size(-1))[:, -1, :]

        # logits = self.classifier(outputs)
        batch_size = outputs.shape[0]
        # print(labels.shape)
        # labels = labels.contiguous().view(-1, 1)
        
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        device = (torch.device('cuda')
                  if outputs.is_cuda
                  else torch.device('cpu'))
        # print(labels)
        # print(labels.T)
        # labels = labels.contiguous().view(-1, 1)
        labels = labels.float()
        labels = labels.view(labels.shape[0], -1)
        # transposed_labels = torch.transpose(labels, 0, 1)
        mask = torch.mm(labels, labels.T).float()
        # print(labels)
        # print(mask)
        outputs = F.softmax(outputs)
        # print(outputs)

        anchor_dot_contrast = torch.div(
            torch.matmul(outputs, outputs.T),
            self.temperature)
        # print('anchor_dot_contrast')
        # print(anchor_dot_contrast)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print(logits_max)
   

        logits = anchor_dot_contrast - logits_max.detach()
        # print('logits=',logits)
        exp_logits = torch.exp(logits)
        
        # print(exp_logits)

        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        # print(num_positives_per_row) 
        # sys.exit()
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)
        # print(denominator)
        log_probs = logits - torch.log(denominator)
        # print(log_probs)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        # print(log_probs)
        # sys.exit()
        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]


        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        # ans = self.out_proj(outputs)
        # ans = F.softmax(ans)
        return loss ,outputs


    # def get_results(self, dataset, batch_size):
    #     '''Given a dataset, return probabilities and labels.'''
    #     self.query += len(dataset)
    #     eval_sampler = SequentialSampler(dataset)
    #     eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)
    #     self.eval()
    #     logits = []
    #     for batch in eval_dataloader:
    #         inputs = batch[0].to("cuda")
    #         label = batch[1].to("cuda")
    #         with torch.no_grad():
    #             lm_loss, logit = self.forward(inputs, label)
    #             # 调用这个模型. 重写了反前向传播模型.
    #             logits.append(logit.cpu().numpy())
    #
    #     logits = np.concatenate(logits, 0)
    #     probs = logits
    #     pred_labels = []
    #     for logit in logits:
    #         pred_labels.append(np.argmax(logit))

        # return probs, pred_labels
