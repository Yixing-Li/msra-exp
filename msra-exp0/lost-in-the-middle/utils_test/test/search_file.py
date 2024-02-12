# from transformers import AutoModelForCausalLM
# import transformers

# print(f'{transformers.__file__}')

import os
path = '/data/data1/yixing/code/msra/msra-exp1/models/flash-attention'
# search = 'mosaicml/mpt-30b-instruct'
# search = 'MPT_PRETRAINED_MODEL_ARCHIVE_LIST'
# search = 'BaseModelOutputWithPast'
#---
# path = '/data/data1/yixing/codes/msra/msra-exp1/lost-in-the-middle'
# search = 'best_subspan_em'
search = 'flash_attn_2_cuda'
search_file = 'flash_attn_2_cuda'

found_dict = {}
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        if search_file in file:
            print(f'!!!!!!!{file_path}!!!!!!!!\n\n\n')
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
    print(f'\n**path+{key[len(path):]}: {item} **\n')