
import os, sys, math, random, time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
from tqdm import tqdm
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils

def ToD(batch, device):
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    if isinstance(batch, Dict):
        for outkey in batch:
            if isinstance(batch[outkey], torch.Tensor):
                batch[outkey] = batch[outkey].to(device)
            if isinstance(batch[outkey], Dict):
                for inkey in batch[outkey]:
                    if isinstance(batch[outkey][inkey], torch.Tensor):
                        batch[outkey][inkey] = batch[outkey][inkey].to(device)
    return batch

def csr_to_pad_tensor(spmat, pad):
    inds_tensor = torch.LongTensor(spmat.indices)
    data_tensor = torch.LongTensor(spmat.data)
    return {'inds': torch.nn.utils.rnn.pad_sequence([inds_tensor[spmat.indptr[i]:spmat.indptr[i+1]] for i in range(spmat.shape[0])], batch_first=True, padding_value=pad),
           'vals': torch.nn.utils.rnn.pad_sequence([data_tensor[spmat.indptr[i]:spmat.indptr[i+1]] for i in range(spmat.shape[0])], batch_first=True, padding_value=0.0)}

def read_sparse_mat(filename, use_xclib=True):
    if use_xclib:
        return xclib.data.data_utils.read_sparse_file(filename)
    else:
        with open(filename) as f:
            nr, nc = map(int, f.readline().split(' '))
            data = []; indices = []; indptr = [0]
            for line in tqdm(f):
                if len(line) > 1:
                    row = [x.split(':') for x in line.split()]
                    tempindices, tempdata = list(zip(*row))
                    indices.extend(list(map(int, tempindices)))
                    data.extend(list(map(float, tempdata)))
                    indptr.append(indptr[-1]+len(tempdata))
                else:
                    indptr.append(indptr[-1])
            score_mat = sp.csr_matrix((data, indices, indptr), (nr, nc))
            del data, indices, indptr
            return score_mat

from xclib.utils.sparse import rank as sp_rank

def _topk(rank_mat, K, inplace=False):
    topk_mat = rank_mat if inplace else rank_mat.copy()
    topk_mat.data[topk_mat.data > K] = 0
    topk_mat.eliminate_zeros()
    return topk_mat

def Recall(rank_intrsxn_mat, true_mat, K=[1,3,5,10,20,50,100]):
    K = sorted(K, reverse=True)
    topk_intrsxn_mat = rank_intrsxn_mat.copy()
    res = {}
    for k in K:
        topk_intrsxn_mat = _topk(topk_intrsxn_mat, k, inplace=True)
        res[k] = (topk_intrsxn_mat.getnnz(1)/true_mat.getnnz(1)).mean()*100.0
    return res

def MRR(rank_intrsxn_mat, true_mat, K=[1,3,5,10,20,50,100]):
    K = sorted(K, reverse=True)
    topk_intrsxn_mat = _topk(rank_intrsxn_mat, K[0], inplace=True)
    rr_topk_intrsxn_mat = topk_intrsxn_mat.copy()
    rr_topk_intrsxn_mat.data = 1/rr_topk_intrsxn_mat.data
    max_rr = rr_topk_intrsxn_mat.max(axis=1).toarray().ravel()
    res = {}
    for k in K:
        max_rr[max_rr < 1/k] = 0.0
        res[k] = max_rr.mean()*100
    return res

def XCMetrics(score_mat, X_Y, inv_prop, disp = True, fname = None, method = 'Method'): 
    X_Y = X_Y.tocsr().astype(np.bool_)
    acc = xc_metrics.Metrics(X_Y, inv_prop)
    xc_eval_metrics = np.array(acc.eval(score_mat, 5))*100
    xc_eval_metrics = pd.DataFrame(xc_eval_metrics)
    
    if inv_prop is None : xc_eval_metrics.index = ['P', 'nDCG']
    else : xc_eval_metrics.index = ['P', 'nDCG', 'PSP', 'PSnDCG']    
    xc_eval_metrics.columns = [(i+1) for i in range(5)]
    
    rank_mat = sp_rank(score_mat)
    intrsxn_mat = rank_mat.multiply(X_Y)
    ret_eval_metrics = pd.DataFrame({'R': Recall(intrsxn_mat, X_Y, K=[10, 20, 100]), 'MRR': MRR(intrsxn_mat, X_Y, K=[10])}).T
    ret_eval_metrics = ret_eval_metrics.reindex(sorted(ret_eval_metrics.columns), axis=1)
        
    df1 = xc_eval_metrics[[1,3,5]].iloc[[0,1,2]].round(2).stack().to_frame().transpose()
    df2 = ret_eval_metrics.iloc[[0,1]].round(2).stack().to_frame().transpose()

    df = pd.concat([df1, df2], axis=1)
    df.columns = [f'{col[0]}@{col[1]}' for col in df.columns.values]
    df.index = [method]

    if disp:
        disp_df = df[['P@1', 'P@5', 'nDCG@1', 'nDCG@5', 'PSP@1', 'PSP@5', 'R@10', 'R@20', 'R@100', 'MRR@10']].round(2)
        print(disp_df.to_csv(sep='\t', index=False))
        print(disp_df.to_csv(sep=' ', index=False))
    if fname is not None:
        if os.path.splitext(fname)[-1] == '.json': df.to_json(fname)
        elif os.path.splitext(fname)[-1] == '.csv': df.to_csv(fname)  
        elif os.path.splitext(fname)[-1] == '.tsv': df.to_csv(fname, sep='\t')  
        else: print(f'ERROR: File extension {os.path.splitext(fname)[-1]} in {fname} not supported')
    return df


# +
class bcolors:
    purple = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    warn = '\033[93m' # dark yellow
    fail = '\033[91m' # dark red
    white = '\033[37m'
    yellow = '\033[33m'
    red = '\033[31m'
    
    ENDC = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'
    reverse = '\033[7m'
    
    on_grey = '\033[40m'
    on_yellow = '\033[43m'
    on_red = '\033[41m'
    on_blue = '\033[44m'
    on_green = '\033[42m'
    on_magenta = '\033[45m'
    
def _c(*args, attr='bold'):
    string = ''.join([bcolors.__dict__[a] for a in attr.split()])
    string += ' '.join([str(arg) for arg in args])+bcolors.ENDC
    return string
