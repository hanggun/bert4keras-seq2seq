import glob
from bert4keras.tokenizers import Tokenizer, load_vocab
from config import config
from tqdm import tqdm

def tokenize_csl():
    token_dict, keep_tokens = load_vocab(
        dict_path=config.token_dict,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict=token_dict, do_lower_case=True)
    data_dir = 'data/csl_10k/'
    files = glob.glob(data_dir+'/*.tsv')
    for file in files:
        filedata = open(file, 'r', encoding='utf-8').read().splitlines()
        target = []
        source = []
        for line in tqdm(filedata, desc=f'process {file}'):
            line = line.split('\t')
            target.append(' '.join(tokenizer.tokenize(line[0], maxlen=256)[1:-1]))
            source.append(' '.join(tokenizer.tokenize(line[1], maxlen=256)[1:-1]))
        with open(file[:-3]+'tok.source', 'w', encoding='utf-8') as f:
            f.write('\n'.join(source))
        with open(file[:-3]+'tok.target', 'w', encoding='utf-8') as f:
            f.write('\n'.join(target))

tokenize_csl()