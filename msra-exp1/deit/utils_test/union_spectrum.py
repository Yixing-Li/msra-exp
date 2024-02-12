import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import os

def get_rank(x):
    layer_head = x[x.find('layer_'):x.find('.png')]
    layer = int(layer_head[len('layer_'):layer_head.find('-head')])
    head  = int(layer_head[layer_head.find('head_')+len('head_'):])
    return layer * 100 + head

def union_image(file_name = None):
    if file_name == None:
        # file_name = '/home/LeiFeng/Yixing/code/msra/msra-exp1/deit/output/deit_base_distilled_patch16_224/double-softmax_inside-exp_inside-clip/spectrum'     
        file_name = '/home/LeiFeng/Yixing/code/msra/msra-exp1/deit/output/deit_base_distilled_patch16_224/baseline-1.24/spectrum'                    
    file_path = os.path.abspath(file_name)     
    filelists = os.listdir(file_path)

    filelists.sort(key = lambda x: get_rank(x))

    m, n = 12, 12
    img_name = file_name[file_name.find('output/')+len('output/') : -len('/spectrum')]
    print(11, img_name,22, file_name.find('output/'), file_name[46:])
    img_name = img_name.replace('/', '--')
    img_name = f'/home/LeiFeng/Yixing/code/msra/msra-exp1/deit/utils_test/union_spectrum/{img_name}'
    img_list = []
    for i in filelists:
        img_list.append(mping.imread(f"{file_name}/{i}"))
    img_temp = []
    for i in range(0,m*n,n):
        img_temp.append(np.concatenate(img_list[i:i+n],axis=1))
    img_end = np.concatenate(img_temp,axis=0)

    print(img_name)
    mping.imsave(f"{img_name}.png",img_end)

def main():
    union_image()

if __name__ == '__main__':
    main()