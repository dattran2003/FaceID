import yaml

class Config:
    def __init__(self, config_file='config/config.yaml'):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
            print(self.config)
            print(type(self.config))

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)
    
    


