# -*- coding: UTF-8 -*-
import os
from load_path import aggregation_lobe, aggregation_gyrus
# 忽略BrainCog引起的警告
ignore_braincog = '-W ignore'

# Change your aggregation type here
for aggregation_type in [aggregation_lobe, aggregation_gyrus]:
    parameters_string = f'--aggregation_type {aggregation_type}'
    exit_code = os.system(f'python {ignore_braincog} SNN.py {parameters_string}')
    if not exit_code == 0:
        print(f'Wrong with {aggregation_type}. Exit Code = {exit_code}')
        exit(exit_code)

# Cleanup unnecessary files and optimize the local repository
os.system('git gc --prune=now')