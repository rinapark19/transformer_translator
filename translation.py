import warnings
warnings.filterwarnings("ignore")

import os
import torch
from torch.autograd import Variable
from util import subsequent_mask
from transformer import Transformer
from transformers import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu

if __name__=="__main__":
  project_dir = '../..'
  checkpoint_path = 'checkpoint_big/transformer-translation-spoken.pth'

  # model setting
  model_name = 'eng-ko-translator'
  vocab_num = 22000
  max_length = 128
  d_model = 512
  head_num = 8
  dropout = 0.1
  N = 6
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

  tokenizer = BertTokenizer(vocab_file='Jeju_dialect/wiki-vocab.txt', do_lower_case=False)
  
  model = Transformer(vocab_num=vocab_num,
                      d_model=d_model,
                      max_seq_len=max_length,
                      head_num=head_num,
                      dropout=dropout,
                      N=N)
  
  if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    global_steps = checkpoint['train_step']

    model.load_state_dict(checkpoint['model_state_dict'])
    print('checkpoints/transformer-translation-spoken.pth loaded')
    model.eval()

    while True:
        input_str = input('입력: ')
        str = tokenizer.encode(input_str)
        pad_len = (max_length - len(str))
        str_len = len(str)
        encoder_input = torch.tensor(str + [tokenizer.pad_token_id]* pad_len)
        encoder_mask = (encoder_input != tokenizer.pad_token_id).unsqueeze(0)

        target = torch.ones(1, 1).fill_(tokenizer.cls_token_id).type_as(encoder_input)
        encoder_output = model.encode(encoder_input,encoder_mask)

        for i in range(max_length - 1):
            lm_logits = model.decode(encoder_output,encoder_mask,target, Variable(subsequent_mask(target.size(1)).type_as(encoder_input.data)))
            prob = lm_logits[:, -1]
            _, next_word = torch.max(prob, dim=1)

            if next_word.data[0] == tokenizer.pad_token_id or next_word.data[0] == tokenizer.sep_token_id:
                #print(f'ko: {input_str} en: {tokenizer.decode(target.squeeze().tolist(),skip_special_tokens=True)}')
                print("원문", input_str)
                print("번역문", tokenizer.decode(target.squeeze().tolist(),skip_special_tokens=True))

                # BLEU Score
                output_sentence = tokenizer.decode(target.squeeze().tolist(),skip_special_tokens=True)
                reference = output_sentence.split()
                print(reference)
                print('enter the label sentence')
                candidate = []
                label_sentence = input().split()
                candidate.append(label_sentence)
                print('calculating bleu score....')
                print('BLEU:', sentence_bleu(references=candidate, hypothesis=reference, weights=(1, 0, 0, 0)))

                break


            target = torch.cat((target[0], next_word))
            target = target.unsqueeze(0)