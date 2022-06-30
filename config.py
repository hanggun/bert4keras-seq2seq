import yaml

config_path = 'yamls/multi30k.yaml'

with open(config_path, "r", encoding='utf-8') as f:
    cfg = yaml.load(f, yaml.SafeLoader)

class Config:
    def __init__(self):
        for key, value in cfg.items():
            setattr(self, key, value)

config = Config()