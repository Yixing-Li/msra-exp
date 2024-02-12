import numpy as np
from tqdm import tqdm
import sys
import torch

def softmax(x, dim = 1):
    max = np.max(
        x, axis=dim, keepdims=True
    )  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(
        e_x, axis=dim, keepdims=True
    )  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

def softmax_0(x, dim = 1):
    # max = np.max(
    #     x, axis=dim, keepdims=True
    # )  # returns max of each row and keeps same dims
    e_x = np.exp(x)  # subtracts each row with its max value
    # sum = np.sum(
    #     e_x, axis=dim, keepdims=True
    # )  # returns sum of each row and keeps same dims
    # f_x = e_x / sum
    return e_x

if False:
    trial_time = 10000
    success_time = 0
    for i in tqdm(range(trial_time)):
        a = np.random.random((5,5))
        a_sm = softmax(a)

        k = 1
        a_sm *= k


        # print(a, '\n')
        # print(a_sm)
        eva, evt = np.linalg.eig(a_sm)

        eva = np.around(eva,3)
        eva = list(eva)
        eva.sort(key = lambda x: abs(x), reverse = True)
        # eva = eva[::-1]
        # print(f'{a_sm}\n\ne-value: {eva}\n\n')

        # print(f'eva[0] == {k}: {eva[0] == k}')
        if eva[0] == k:
            success_time += 1

    print(f'{success_time} / {trial_time} = {success_time / trial_time}')

if False:
    a = np.random.random((3,3))
    a_sum = np.sum(a, axis= -1)
    a_sum_sm = softmax(a_sum, dim=-1)
    a_sum_sm_diag = np.diag(a_sum_sm)
    print(f'{a}\n\n{a_sum.shape}\n\n{a_sum_sm.shape}\n\n{a_sum_sm_diag.shape}')
    print(f'\n\n******\n{a_sum_sm}\n\n{a_sum_sm_diag}')
    # print(f'{a}\n\n{a_sum}\n\n')
    a_sm = softmax_0(a)

    a_sm = np.dot(a_sum_sm_diag, a_sm)

    k = 1
    a_sm *= k


    # print(a, '\n')
    # print(a_sm)
    eva, evt = np.linalg.eig(a_sm)

    eva = np.around(eva,3)
    eva = list(eva)
    eva.sort(key = lambda x: abs(x), reverse = True)
    # eva = eva[::-1]
    print(f'{a_sm}\n\ne-value: {eva}\n\n')

if False:
    a = torch.Tensor([[[1, 2, 3], [3, 4, 3]], [[4,3, 3],[4,4,3 ]]]);   print(f'a:{a.shape}\n{a}\n')
    a_sum = torch.sum(a, axis= -1);       print(f'a_sum:{a_sum.shape}\n{a_sum}\n')
    a_sum_sm = a_sum.softmax(dim=-1);     print(f'a_sum_sm:{a_sum_sm.shape}\n{a_sum_sm}\n')
    a_sum_sm_diag = torch.diag(a_sum_sm); print(f'a_sum_sm_diag:{a_sum_sm_diag.shape}\n{a_sum_sm_diag}\n')

    b_1 = torch.matmul(a_sum_sm, a);  print(f'{"*"*10}\n a_sum_sm:\n{a_sum_sm}\n a:\n{a}\nb_1:\n{b_1}')
    
    # print(f'{a}\n\n{a_sum.shape}\n\n{a_sum_sm.shape}\n\n{a_sum_sm_diag.shape}')
    # print(f'\n\n******\n{a_sum_sm}\n\n{a_sum_sm_diag}')
    # print(f'{a}\n\n{a_sum}\n\n')
    a_sm = softmax_0(a)

    a_sm = np.dot(a_sum_sm_diag, a_sm)

if False:
    # 创建一个示例的3维张量
    a = torch.Tensor([[[1, 2], [3, 4]], [[4,3],[4,4 ]]]);   print(f'a:{a.shape}\n{a}\n')
    a_sum = a.sum(axis= -1);       print(f'a_sum:{a_sum.shape}\n{a_sum}\n')
    tensor_matrix = a_sum.softmax(dim=-1);     print(f'tensor_matrix:{tensor_matrix.shape}\n{tensor_matrix}\n')
    # tensor_matrix = torch.randn((2, 3, 4))

   
    # # 创建一个示例的2维张量
    # tensor_matrix = torch.randn((2, 3))

    # 获取最后一个维度的大小, # 将最后一个维度扩展为对角矩阵
    expanded_tensor = torch.diag_embed(tensor_matrix)

    print(expanded_tensor.shape, expanded_tensor)
    result = torch.matmul(expanded_tensor, a)
    print(result)
    sum_result = torch.sum(result, dim = [1, 2])
    print(sum_result)

if False:
    # 创建一个示例的张量
    tensor_example = torch.tensor([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]])
    row_sums = torch.sum(tensor_example, dim=1, keepdim=True)  # 求每一行的和
    print(f'原来的归一化向量：{row_sums}')
    total_sum = torch.sum(row_sums)
    row_sums /= total_sum
    print(f'归一化向量：{row_sums}, {total_sum}')

    tensor_example = tensor_example.softmax(dim = -1)
    print(f'softmax后的向量:{tensor_example}')
    # 对每一行进行归一化
    normalized_tensor = tensor_example * row_sums  # 每一行乘以该行的和
    print(f'归一化后的softmax的向量:{normalized_tensor}')

    # 将每一行的和除以所有行的和
    # print(normalized_tensor)

    '''
    # this version is correct!!
    row_sums = torch.sum(tensor_example, dim=1, keepdim=True)  
    total_sum = torch.sum(row_sums)
    row_sums /= total_sum

    tensor_example = tensor_example.softmax(dim = -1)
    normalized_tensor = tensor_example * row_sums  
    '''

    '''
    # this version is wrong.
    sum_attn = attn.sum(axis= -1)
    sumnormed_attn = sum_attn.softmax(dim=-1)
    diag_sumnormed_attn = torch.diag_embed(sumnormed_attn)
    attn = torch.matmul(diag_sumnormed_attn, attn)
    '''

if False:
    tensor_example = torch.randn((12, 12, 3, 3))

    row_sums = torch.sum(tensor_example, dim=(-1), keepdim=True)  # 求最后两个维度的和
    # total_sum = torch.sum(row_sums, dim=(-1, -2), keepdim=True)
    # row_sums2 = row_sums / total_sum
    row_sums2 = row_sums.softmax(dim=-2)
    # print(f'{total_sum.shape},' )
    print(f'{row_sums.shape}, {row_sums2.shape}')

    tensor_example = tensor_example.softmax(dim=-1)
    normalized_tensor = tensor_example * row_sums2  # 每个元素除以对应行的和


    # 打印结果
    print("原始张量:")
    print(tensor_example[0, 0])

    print("\n每一行的和:")
    print(row_sums[0, 0])
    print(row_sums2[0, 0])

    print("\n每一行归一化后的结果:")
    print(normalized_tensor[0, 0])

if True:
    inside_exp = False
    outside_exp = False

    attn = torch.randn((12, 12, 3, 3))
    # get org_attn -> calculate sum_attn -> do attn softmax -> do attn double-norm
    if not inside_exp:
        sum_attn = torch.sum(attn, dim = -1, keepdim=True) 
    else:
        exp_attn = attn.exp()
        sum_attn = torch.sum(exp_attn, dim = -1, keepdim=True) 

    if not outside_exp: 
        total_sum = torch.sum(sum_attn, dim=-2, keepdim=True)
        normed_sum_attn = sum_attn / total_sum
    else:
        normed_sum_attn = sum_attn.softmax(dim = -2)

    attn_sftmax = attn.softmax(dim=-1)
    normalized_tensor = normed_sum_attn * attn_sftmax

    # 打印结果
    print("原始张量:")
    print(attn[0, 0])

    print("\n每一行的和:")
    print(sum_attn[0, 0])
    print(normed_sum_attn[0, 0])

    print("\n原始张量做softmax:")
    print(attn_sftmax[0, 0])

    print("\n每一行归一化后的结果:")
    print(normalized_tensor[0, 0])
    print(normalized_tensor[0, 0].sum())

if False:
    a = torch.Tensor([[1, 2], [0, 3]])
    b = a.exp()
    print(a)
    print(b)