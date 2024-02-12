import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import math

pre_align = 0
def np_add(add1, add2):
    assert pre_align != 0
    try:
        result = add1 + add2
        return add1 + add2
    except:
        div = add1.shape[0] - add2.shape[0]
        if div > 0:
            small = add2
            big = add1
        else:
            small = add1
            big = add2
        np_append = np.zeros( int(abs(div)) )
        if pre_align > 0:
            small = np.append(small, np_append)
        else:
            small = np.append(np_append, small)
        result = small + big
        return result

def main():
    global pre_align
    # print(plt.style.available)
    plt.style.use('seaborn-v0_8')

    # sys.path.append('')
    cal_average = True
    absolute = False
    pre_align = -1

    test_docu_list = [] # range(0,11)
    test_block_list = [47]

    cal_average_str = 'average' if cal_average else ''
    absolute_str = 'absolute' if absolute else ''
    conn = '-' if (absolute and cal_average) else ''

    # 2023_12_20-12_00
    # 2023_12_18-14_19
    exp = 'attn_score-nq_open-docu_30-gold_4-m_3_i/2023_12_18-14_19'
    exp_data_dir = f'/data/data1/yixing/code/msra/msra-exp1/lost-in-the-middle/utils_test/output/{exp}/attn_score'
    out_dir = f'/data/data1/yixing/code/msra/msra-exp1/lost-in-the-middle/utils_test/output/{exp}/attn_visualization-{cal_average_str}{conn}{absolute_str}'
    Path(out_dir).mkdir(exist_ok = True, parents = True)


    for root, dirs, files in os.walk(exp_data_dir):
        if cal_average:
            attn_score_sum = None
            attn_cnt = 0
            got_attn_score = False
            attn_min_x = math.inf
        for dir in sorted(dirs, key = lambda x: int(x[x.find('-')+1:]) ):
            if 'prompt' in dir:
                # get the prompt id.
                prompt_id = int(dir[dir.find('-')+1:])
                if (len(test_docu_list) > 0) and (prompt_id not in test_docu_list):
                    continue
                
                for root_pt, dirs_pt, files_pt in os.walk(os.path.join(root, dir)):
                    for file_pt in sorted(files_pt, key = lambda x: int(x[x.find('-')+1:x.find('.npy')])):
                        # get the prompt_id and the block_id.
                        block_id = int(file_pt[file_pt.find('-')+1:file_pt.find('.npy')])
                        if (len(test_block_list) > 0 ) and ( block_id not in test_block_list):
                            continue

                        # get the attn score.
                        if cal_average:
                            if block_id == 47:
                                got_attn_score = True
                                attn_score = np.load(os.path.join(root_pt, file_pt))
                                attn_score = np.squeeze(attn_score, 0)
                                if absolute:
                                    attn_score = np.absolute(attn_score)

                                attn_min_x = min(attn_min_x, attn_score.shape[0])
                                
                                # print(attn_score.min())
                                if attn_cnt == 0:
                                    attn_score_sum = attn_score
                                    attn_cnt += 1
                                else:
                                    # attn_score_sum += attn_score
                                    attn_score_sum = np_add(attn_score_sum, attn_score)
                                    attn_cnt += 1
                        else:
                            attn_score = np.load(os.path.join(root_pt, file_pt))
                            attn_score = np.squeeze(attn_score, 0)
                            if absolute:
                                attn_score = np.absolute(attn_score)
                            x = np.arange(1, (attn_score.shape[-1]+1))
                            plt.bar(x, attn_score)
                            plt.savefig(f'{out_dir}/prompt_{prompt_id}-block_{block_id}.png')
                            plt.clf()
        if cal_average and got_attn_score:
            attn_score_sum /= attn_cnt
            x = np.arange(1, (attn_score_sum.shape[-1]+1))

            align_str = 'pre_align' if pre_align > 0 else 'post_align'
            valid_x_bound = attn_min_x if pre_align > 0 else ( len(x)-attn_min_x )
            
            plt.bar(x, attn_score_sum)
            plt.axvline(x=valid_x_bound, color = 'r')

            plot_str = f'Coordinates of Extreme Values:\nmax: ({attn_score_sum.argmax()}, {attn_score_sum.max():.2f})\nmin: ({attn_score_sum.argmin()}, {attn_score_sum.min():.2f})'
            plot_cordinate = (4000, attn_score_sum.max()-10)
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
            plt.text(plot_cordinate[0],plot_cordinate[1], plot_str, ha="center", va="center", size=10, bbox=bbox_props)

            plt.xlabel('token index')
            plt.ylabel('attn score')
            save_fig_dir = f'{out_dir}/avg_prompt_attn-{align_str}--block_47.png'
            plt.savefig(save_fig_dir)
            print(f'saved img at: {save_fig_dir}')
            plt.clf()

                                      
if __name__ == '__main__':
    main()
'''

'''