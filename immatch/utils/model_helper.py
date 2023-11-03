import os
import yaml
import immatch

def parse_model_config(config, benchmark_name, root_dir='.'):    
    config_file=get_yaml_path(config)
    #config_file = f'{root_dir}/configs/{config}.yml'
    with open(config_file, 'r') as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)[benchmark_name]
        
        # Update pretrained model path
        if 'ckpt' in model_conf and root_dir != '.':
            model_conf['ckpt'] = os.path.join(root_dir, model_conf['ckpt'])
            if 'coarse' in model_conf and 'ckpt' in model_conf['coarse']:
                model_conf['coarse']['ckpt'] = os.path.join(
                    root_dir, model_conf['coarse']['ckpt']
                )
    return model_conf

def init_model(config, benchmark_name, root_dir='.'):

    model_conf = parse_model_config(config, benchmark_name, root_dir)
    # Initialize model
    class_name = model_conf['class']
    model = immatch.__dict__[class_name](model_conf)
    print(f'Method:{class_name} Conf: {model_conf}')
    return model, model_conf



from importlib import resources
import importlib.resources as pkg_resources
from pathlib import Path

def get_yaml_path(config):
    return pkg_resources.files('immatch.configs')/Path(config+'.yml')


def get_parent_directory_of_package():
    # 找到包内任何模块的路径，这里选择 __init__.py 文件作为参考点
    # 这里使用 'your_package' 因为我们要找这个包的父目录
    with resources.path('immatch', '__init__.py') as init_file:
            # 获取your_package所在的目录（init_file的父目录）
            package_directory = init_file.parent
        # 获取上一级目录
    parent_directory = package_directory.parent
    return parent_directory

    
def init_matcher(config,benchmark_name):
    root_dir=get_parent_directory_of_package()
    model_conf=parse_model_config(config, benchmark_name, root_dir)

    class_name = model_conf['class']
    model = immatch.__dict__[class_name](model_conf)
    return model, model_conf
