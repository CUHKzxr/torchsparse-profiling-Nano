import csv
import numpy as np
import pandas as pd
import torch
from torch import nn

from typing import List

import torchsparse
from torchsparse import SparseTensor
from torchsparse.backbones import SparseResNet21D, SparseResUNet42
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.nn import Conv3d, BatchNorm
from torch.profiler import profile, record_function, ProfilerActivity

import copy
import os
import onnx
import onnx.numpy_helper

import cProfile
import sys
import time

def load_feats(filename):
    # 以二进制模式打开文件并读取内容
    with open(filename, 'rb') as file:
        # 读取 shape
        rows = int.from_bytes(file.read(8), byteorder='little', signed=True)
        cols = int.from_bytes(file.read(8), byteorder='little', signed=True)

        # 读取特征数据
        data = np.frombuffer(file.read(rows * cols * 4), dtype=np.float32)
    
    # 将一维数组转换为二维数组
    feats = np.reshape(data, (rows, cols))

    return feats

class nameTree:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children
        
    def setChildren(self, obj):
        self.children = obj

def name_mapping(onnx_names:List[str], module_names:List[str] ,mapdict):
    onnx_name = onnx_names[0]
    name_tree = mapdict[onnx_name]
    name = name_tree.name
    if(not name is None):
        module_names.append(name)
    if(name_tree.children is None):
        return module_names
    elif(isinstance(name_tree.children, dict)):
        return name_mapping(onnx_names[1:], module_names, name_tree.children)
    elif(isinstance(name_tree.children, str)):
        module_names.insert(0, name_tree.children)
        return module_names
        

def resnet_dict_():
    dict_res_main = {'conv3d1':nameTree('0'), 'batchnorm1':nameTree('1'), 'relu':nameTree('2'), 
                     'conv3d2':nameTree('3'), 'batchnorm1':nameTree('4')}
    dict_res_shortcut = {'conv3d':nameTree('0'), 'batchnorm':nameTree('1')}
    dict_resblock = {'main':nameTree('main', dict_res_main), 'shortcut':nameTree('shortcut', dict_res_shortcut),
                     'relu':nameTree('relu')}
    dict_convblock = {'conv3d':nameTree('0'), 'batchnorm':nameTree('1'), 'relu':nameTree('2')}
    dict_block = {'0':nameTree('0', dict_convblock), '1': nameTree('1', dict_resblock), '2': nameTree('2', dict_resblock)}
    dict_resnet = {}
    for i in range(5):
        dict_resnet[str(i)] = nameTree(str(i), dict_block)
    return dict_resnet

def resnet_dict():
    dict_res_main = {'conv3d1':nameTree('0'), 'batchnorm1':nameTree('1'), 'relu':nameTree('2'), 
                     'conv3d2':nameTree('3'), 'batchnorm1':nameTree('4')}
    dict_res_shortcut = {'conv3d':nameTree('0'), 'batchnorm':nameTree('1')}
    dict_resblock = {'main':nameTree('main', dict_res_main), 'shortcut':nameTree('shortcut', dict_res_shortcut),
                     'relu':nameTree('relu')}
    dict_convblock = {'conv3d':nameTree('0'), 'batchnorm':nameTree('1'), 'relu':nameTree('2')}
    dict_list = {}
    for i in  range(13):
        block_idx, layer_idx = divmod(i, 3)
        value_name = str(block_idx) + '.' + str(layer_idx)
        if(layer_idx == 0 ):
            dict_list[str(i)] = nameTree(value_name, copy.deepcopy(dict_convblock))
        else:
            dict_list[str(i)] = nameTree(value_name, copy.deepcopy(dict_resblock))
    dict_resnet = {'modulelist':nameTree(None, dict_list)}
    return dict_resnet

def unet_dict_():
    dict_res_main = {'conv3d1':nameTree('0'),  
                     'conv3d2':nameTree('3'), }
    dict_res_main_ = {'conv3d1':nameTree('0', 'decoders.0'), 'conv3d1_1':nameTree('0', 'decoders.1'), 
                     'conv3d1_2':nameTree('0', 'decoders.2'), 'conv3d1_3':nameTree('0', 'decoders.3'), 
                     'conv3d2':nameTree('3', 'decoders.0'), 'conv3d2_1':nameTree('3', 'decoders.1'), 
                     'conv3d2_2':nameTree('3', 'decoders.2'), 'conv3d2_3':nameTree('3', 'decoders.3'), }
    dict_res_shortcut = {'conv3d':nameTree('0'), }
    dict_res_shortcut_ = {'conv3d':nameTree('0', 'decoders.0'), 'conv3d_1':nameTree('0', 'decoders.1'), 
                          'conv3d_2':nameTree('0', 'decoders.2'), 'conv3d_3':nameTree('0', 'decoders.3'), }
    dict_resblock = {'main':nameTree('main', dict_res_main), 'shortcut':nameTree('shortcut', dict_res_shortcut), 
                     'relu':nameTree('relu')}
    dict_resblock_ = {'main':nameTree('main', dict_res_main_), 'shortcut':nameTree('shortcut', dict_res_shortcut_), 
                     'relu':nameTree('relu')}
    dict_convblock = {'conv3d':nameTree('0'), 'batchnorm':nameTree('1'), 'relu':nameTree('2')}
    dict_convblock_t = {'conv3d':nameTree('0','decoders.0'), 'conv3d_1':nameTree('0','decoders.1'), 
                        'conv3d_2':nameTree('0','decoders.2'), 'conv3d_3':nameTree('0','decoders.3'),}
    dict_fuseblock = {'resblock1':nameTree('0', copy.deepcopy(dict_resblock_)), 
                      'resblock2':nameTree('1', copy.deepcopy(dict_resblock_))}
    dict_encoder = {'convblock':nameTree('0', copy.deepcopy(dict_convblock)), 
                    'resblock1':nameTree('1', copy.deepcopy(dict_resblock)), 
                    'resblock2':nameTree('2', copy.deepcopy(dict_resblock))}
    dict_decoder = {'upsample':nameTree('upsample', copy.deepcopy(dict_convblock_t)), 
                    'fuse':nameTree('fuse', copy.deepcopy(dict_fuseblock))}
    dict_enocders = {'0':nameTree('0', copy.deepcopy(dict_encoder)), 
                     '1':nameTree('1', copy.deepcopy(dict_encoder)), 
                     '2':nameTree('2', copy.deepcopy(dict_encoder)), 
                     '3':nameTree('3', copy.deepcopy(dict_encoder))}
    dict_stem = {'conv3d1':nameTree('0'), 'batchnorm1':nameTree('1'), 'relu1':nameTree('2'), 
                 'conv3d2':nameTree('3'), 'batchnorm2':nameTree('4'), 'relu2':nameTree('5')}
    dict_unet = {'stem':nameTree('stem', dict_stem), 
               'encoders':nameTree('encoders',dict_enocders), 
               'upsample':nameTree('upsample', copy.deepcopy(dict_convblock_t)), 
                'fuse':nameTree('fuse', copy.deepcopy(dict_fuseblock))}
    return dict_unet

  
def unet_dict():
    dict_res_main = {'conv3d1':nameTree('0'), 'batchnorm1':nameTree('1'), 'relu':nameTree('2'), 
                     'conv3d2':nameTree('3'), 'batchnorm1':nameTree('4')}
    dict_res_shortcut = {'conv3d':nameTree('0'), 'batchnorm':nameTree('1')}
    dict_resblock = {'main':nameTree('main', dict_res_main), 'shortcut':nameTree('shortcut', dict_res_shortcut), 
                     'relu':nameTree('relu')}
    dict_convblock = {'conv3d':nameTree('0'), 'batchnorm':nameTree('1'), 'relu':nameTree('2')}
    dict_convblock_t = copy.deepcopy(dict_convblock)
    dict_fuseblock = {'resblock1':nameTree('0', copy.deepcopy(dict_resblock)), 
                      'resblock2':nameTree('1', copy.deepcopy(dict_resblock))}
    dict_encoder = {'convblock':nameTree('0', copy.deepcopy(dict_convblock)), 
                    'resblock1':nameTree('1', copy.deepcopy(dict_resblock)), 
                    'resblock2':nameTree('2', copy.deepcopy(dict_resblock))}
    dict_decoder = {'upsample':nameTree('upsample', copy.deepcopy(dict_convblock_t)), 
                    'fuse':nameTree('fuse', copy.deepcopy(dict_fuseblock))}
    dict_enocders = {'0':nameTree('0', copy.deepcopy(dict_encoder)), 
                     '1':nameTree('1', copy.deepcopy(dict_encoder)), 
                     '2':nameTree('2', copy.deepcopy(dict_encoder)), 
                     '3':nameTree('3', copy.deepcopy(dict_encoder))}
    dict_decoders = {'0':nameTree('0', copy.deepcopy(dict_decoder)), 
                     '1':nameTree('1', copy.deepcopy(dict_decoder)), 
                     '2':nameTree('2', copy.deepcopy(dict_decoder)), 
                     '3':nameTree('3', copy.deepcopy(dict_decoder))}
    dict_stem = {'conv3d1':nameTree('0'), 'batchnorm1':nameTree('1'), 'relu1':nameTree('2'), 
                 'conv3d2':nameTree('3'), 'batchnorm2':nameTree('4'), 'relu2':nameTree('5')}
    dict_unet = {'stem':nameTree('stem', dict_stem), 
               'encoders':nameTree('encoders',dict_enocders), 
               'decoders':nameTree('decoders', dict_decoders)}
    return dict_unet

def read_weight_from_onnx(module:torch.nn.Module, f, dict, device):
    model_dict = {}
    for name, model in module.named_modules():
        iter = model.named_modules()
        next(iter, None)
        if(next(iter, None) == None):
            model_dict[name]=model
        
    onnx_module = onnx.load(f)
    for initializer in onnx_module.graph.initializer:
        name = initializer.name
        # print(name)
        names = name.split('.')
        names_ = []
        name_mapping(names, names_, dict)
        name_ = '.'.join(names_)
        model = model_dict[name_]
        data_tensor = torch.from_numpy(np.copy(onnx.numpy_helper.to_array(initializer))).to(device)
        # print(name_,':',data_tensor.shape)
        if(names[-1] == 'kernel'):
            model.kernel.data = data_tensor
        elif(names[-1] == 'weight'):
            model.weight.data = data_tensor
        elif(names[-1] == 'bias'):           
            model.bias.data = data_tensor
        elif(names[-1] == 'running_var'):           
            model.running_var.data = data_tensor
        else:
            raise RuntimeError('invalid initializer input name:'+name_)

def statis(list , path, model_type, index):
    if model_type is SparseResUNet42:
        size = 42
    else :
        size = 21
    backend_df = pd.read_csv('/home/nano/torchsparse/data/backend_time.csv',header=None)
    backend = torch.tensor(backend_df.values,dtype=torch.long)
    cur_list = list[-size:]
    cur_list = torch.tensor(cur_list)
    cur_backend = backend[-size:]
    raito = float(1e-9)
    sum_list = torch.sum(cur_list,0)
    sum_backend = torch.sum(cur_backend,0)
    return sum_list[0].item(), sum_backend[0].item() * raito, sum_backend[1].item() * raito, \
                sum_backend[2].item() * raito, sum_list[1].item()

def statis_(list , path, model_type, index):
    if model_type is SparseResUNet42:
        size = 42
    else :
        size = 21
    cur_list = list[-size:]
    cur_list = torch.tensor(cur_list)
    raito = float(1e-9)
    sum_list = torch.sum(cur_list,0)
    return sum_list[0].item(), -1, -1, -1, sum_list[1].item()


@torch.no_grad()
def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    file_path = './data/backend_time.csv'
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"文件 {file_path} 已成功删除")
        except PermissionError:
            print(f"无权限删除文件 {file_path}")
        except Exception as e:
            print(f"删除文件 {file_path} 发生错误：{str(e)}")
    coords_df = pd.read_csv('/home/nano/torchsparse/data/coords.csv',header=None)
    coords = torch.tensor(coords_df.values,dtype=torch.int).to(device)
    feats_df = pd.read_csv('/home/nano/torchsparse/data/feats.csv', header=None)
    feats = torch.tensor(feats_df.values, dtype=torch.float32)
    input = SparseTensor(coords=coords, feats=feats).to(device)
    
    for backbone in [SparseResNet21D]:
        print(f'{backbone.__name__}:')
        model: nn.Module = backbone(in_channels=4, width_multiplier=1.0)
        if(backbone is SparseResUNet42):
            read_weight_from_onnx(model, "/home/nano/torchsparse/data/unet_v2.onnx", unet_dict(), device)
        else:
            read_weight_from_onnx(model, "/home/nano/torchsparse/data/resnet_v2.onnx",resnet_dict_(), device)
        model = model.to(device).eval()
        
        # warm up
        for i in range(10):
            input.cmaps = {}
            input.kmaps = {}
            outputs = model(input)
            torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        
        epoch = 10
        
        results = np.zeros([epoch,2])
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for i in range(epoch):
            input.cmaps = {}
            input.kmaps = {}
            torch.cuda.synchronize(device)
            pre = time.perf_counter()
            starter.record()
            outputs = model(input)
            torch.cuda.synchronize(device)
            post = time.perf_counter()
            ender.record()
            inf_t = post-pre
            _time = starter.elapsed_time(ender) / 1000
            results[i] = [inf_t,_time]
            print("===============================================")
            print("total:", inf_t, ",", _time)
            
        print("============================================")
        print("average:")
        print(np.mean(results,axis=0))
        torch.cuda.synchronize(device)
        results = np.zeros([epoch,2])
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True,) as prof:
            torchsparse.backends.profiling=True
            for i in range(1):
                input.cmaps = {}
                input.kmaps = {}
                torch.cuda.synchronize(device)
                pre = time.perf_counter()
                starter.record()
                outputs = model(input)
                torch.cuda.synchronize(device)
                post = time.perf_counter()
                ender.record()
                # mapping, gather, matmul, scatter, conv = statis_(torchsparse.backends.test_time, './data/backend_time.csv', backbone, 0) 
                # mapping, gather, matmul, scatter, conv = -1, -1, -1, -1, -1,
                inf_t = post-pre
                _time = starter.elapsed_time(ender) / 1000
                # results[i] = [inf_t,mapping, gather, matmul, scatter, conv, inf_t-conv]
                print("===============================================")
                print("total:", inf_t, _time)
                # print("mapping:", mapping, ", ", mapping/inf_t*100, "%")
                # print("gather:", gather, ", ", gather/inf_t*100, "%")
                # print("matmul:", matmul, ", ", matmul/inf_t*100, "%")
                # print("scatter:", scatter, ", ", scatter/inf_t*100, "%")
                # print("conv:", conv, ", ", conv/inf_t*100, "%")
                # print("other:", inf_t-conv, ", ", (inf_t-conv)/inf_t*100, "%")
            print("============================================")
            print("average:")
            print(np.mean(results,axis=0))
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
        prof.export_chrome_trace("./data/trace.json")
            # print(inf_t)
            # print(mapping)
            # print(mapping/inf_t*100, "%")
            # print(gather)
            # print(gather/inf_t*100, "%")
            # print(matmul)
            # print(matmul/inf_t*100, "%")
            # print(scatter)
            # print(scatter/inf_t*100, "%")
            # print(conv)
            # print(conv/inf_t*100, "%")
            # print(inf_t-conv)
            # print((inf_t-conv)/inf_t*100, "%")

if __name__ == '__main__':
    main()
    # profiler = cProfile.Profile()
    # profiler.enable()
    # pre = time.process_time()
    # inf = main()
    # post = time.process_time()
    # print("total time:", post-pre)
    # print(inf/(post-pre))
