import configparser
import re
from pathlib import Path
from collections import defaultdict

#ini_file = Path(__file__).parent / 'config.ini'
ini_file = '/config.ini'

def validate_ini_file(file_path):
    """Validate the config.ini file: look for existence of sections and keys, and check for valid values."""
    config = configparser.ConfigParser()
    if not Path(file_path).exists():
        raise FileNotFoundError(f'Config file {file_path} not found.')
    config.read(file_path)
    config_dict = defaultdict(dict)
    # find all sections with comma sep values and convert them to list else just copy the values into dict:
    for section in config.sections():
        for key in config[section]:
            if ',' in config[section][key]:
                config_dict[section][key] = [i.strip() for i in config[section][key].split(',')]
            elif re.match(r'^-?\d+(?:\.\d+)?$', config[section][key]):
                # look for values that contain only numbers and convert them into int or float:
                config_dict[section][key] = int(config[section][key]) if '.' not in config[section][key] else float(config[section][key])
            elif config[section][key].lower() == 'true' or config[section][key].lower() == 'false':
                # look for values that contain only boolean values and convert them into bool:
                config_dict[section][key] = config[section].getboolean(key)
            else:
                config_dict[section][key] = config[section][key]
    search = config['search']
    # match re.ignorecase for search_method:
    assert re.match(r'cognitivesearch|local|pinecone', search['search_method'], re.IGNORECASE)
    model = config['model']
    assert re.match(r'gpt-3\.?5-turbo|gpt4all|bard|gpt-4', model['model_name'], re.IGNORECASE)
    return config_dict

if __name__ == '__main__':
    validate_ini_file(ini_file)