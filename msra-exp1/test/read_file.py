# from transformers import AutoModelForCausalLM
# import transformers

# print(f'{transformers.__file__}')

import os
# path = '/home/LeiFeng/yixing/conda/anaconda3/envs/lost-in-the-middle/lib/python3.9/site-packages/transformers'
# search = 'mosaicml/mpt-30b-instruct'
# search = 'MPT_PRETRAINED_MODEL_ARCHIVE_LIST'
# search = 'BaseModelOutputWithPast'
#---
path = '/home/LeiFeng/Yixing/code/msra/msra-exp1'
search = 'stopping training'

found_dict = {}
for root, dirs, files in os.walk(path):
    for file in files:
        if 'checkpoint' in file:
            continue
        if 'read_file.py' in file:
            continue
        file_path = os.path.join(root, file)
        if '__pycache__' in file_path:
            continue
        with open(file_path, 'rb') as f:
            lines = f.readlines()
            for ith, line in enumerate(lines):
                if search in str(line):
                    # print(f'\n**{ith+1}: {file_path[len(path):]}**\n')
                    file_info = found_dict.get(file_path, [])
                    file_info.append(ith+1)
                    found_dict[file_path] = file_info

for key, item in found_dict.items():
    print(f'\n**{key[len(path):]}: {item} **\n')