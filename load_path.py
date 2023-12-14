# -*- coding: UTF-8 -*-
import os
 
# 简单文件操作 
path_join = lambda root, leaf: os.path.join(root, leaf)
# 将root路径下含有字段string的所有文件的路径生成一个列表
select_path_list = lambda root, string: [path_join(root, label) for label in os.listdir(root) if string in label]
# 检查path文件夹是否存在 如果不存在则创建一个
check_path_or_create = lambda path: os.makedirs(path) if not os.path.exists(path) else None

# 文件路径 
DEPRESSION_PATH = path_join('..', 'depression_ds002748')
SUBJECTS_PATH = select_path_list(DEPRESSION_PATH, 'sub')
SUBJECTS_FUNC_PATH = [select_path_list(x, 'func') for x in SUBJECTS_PATH]
SUBJECTS_ANAT_PATH = [select_path_list(x, 'anat') for x in SUBJECTS_PATH]
PARICIPANTS_INFO = path_join(DEPRESSION_PATH, 'participants.tsv')
PARICIPANTS_INFO_JSON = path_join('..', 'participants_information.json')
CONNECTION_MATRIX = path_join('..', 'connection_matrix')
check_path_or_create(CONNECTION_MATRIX)
BNA_PATH = path_join('.', 'BN_Atlas')
BNA_MATRIX_PATH = path_join(BNA_PATH, 'BNA_matrix_binary_246x246.csv')
BNA_SUBREGION_PATH = path_join(BNA_PATH, 'subregions.json')
YAML_PATH = path_join('.', 'config.yaml')

# 聚合方式 
aggregation_lobe = 'lobe'
aggregation_gyrus = 'gyrus'
aggregation_not = 'not'