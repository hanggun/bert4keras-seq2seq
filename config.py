import yaml

config_path = 'yamls/multi30k.yaml'
is_train = False

with open(config_path, "r", encoding='utf-8') as f:
    cfg = yaml.load(f, yaml.SafeLoader)

class Config:
    def __init__(self, is_train=True):
        for key, value in cfg.items():
            if key == 'train':
                if is_train:
                    for key_train, value_train in value.items():
                        setattr(self, key_train, value_train)
                continue
            if key == 'evaluate':
                if not is_train:
                    for key_eval, value_eval in value.items():
                        setattr(self, key_eval, value_eval)
                continue
            setattr(self, key, value)

config = Config(is_train=is_train)