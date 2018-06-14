import configparser
import os



def get_project_root(rel = ''):
    cfg_path = os.path.realpath(rel + 'config.ini')
    config = configparser.ConfigParser()
    config.read(cfg_path)
    print(cfg_path)

    project_root = config.get('main', 'PROJECT_ROOT_PATH')
    return project_root

def data_path(rel = ''):
    cfg_path = os.path.realpath(rel + 'config.ini')
    config = configparser.ConfigParser()
    config.read(cfg_path)

    data_path = config.get('main', 'DATA_PATH')
    return data_path
