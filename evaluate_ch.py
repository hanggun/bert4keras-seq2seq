from train import seq2seq_model, de_tokenizer, en_tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
import numpy as np
from snippets import compute_metrics, metric_keys
from tqdm import tqdm
from config import config


class AutoSeq(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        label_ids = output_ids
        return self.last_token(seq2seq_model).predict([token_ids, output_ids, label_ids])

    def generate(self, text, topk=1):
        text = ['[CLS]'] + text.split() + ['[SEP]']
        token_ids = np.array([en_tokenizer.tokens_to_ids(text)])
        output_ids = self.beam_search(token_ids,
                                      topk=topk)  # 基于beam search
        return de_tokenizer.ids_to_tokens(output_ids)

autoseq = AutoSeq(start_id=de_tokenizer._token_start_id, end_id=de_tokenizer._token_end_id, maxlen=config.max_len)
en_data = open(config.test_source_dir, 'r', encoding='utf-8').read().splitlines()
de_data = open(config.test_target_dir, 'r', encoding='utf-8').read().splitlines()

def process_tokens(tokens):
    out_tokens = ''
    for token in tokens:
        if token[:2] == '##':
            out_tokens += token[-2:]
        else:
            out_tokens += ' ' + token
    return out_tokens.split()

total_metrics = {k: 0.0 for k in metric_keys}
total_len = 0
pbar = tqdm(zip(en_data, de_data))
for en, de in pbar:
    total_len += 1
    pred = autoseq.generate(en, topk=4)
    pred = process_tokens(pred)[1:-1]
    true = process_tokens(de.split())
    metrics = compute_metrics(pred, true)
    for k, v in metrics.items():
        total_metrics[k] += v
    pbar.set_description(f"bleu {total_metrics['bleu']/total_len}")
print({k: v / total_len for k, v in total_metrics.items()})