
# 每组实验1h 1s放映10张
# 我只预处理了训练集：
# 训练集：1854 concept 各放映12张，不重复放映
#测试集：\code_and_figure\test_images.csv  200个concept各挑了1张，重复放映 12次

#且 先放映训练集 再紧接着放映测试集

#注意sub-01的tsv、csv文件特殊，不含有teststimnumber等部分标签，可能是作者一开始的代码没记录这些信息，从sub2开始记录
# 注意sub-06的tsv、csv文件特殊，不含有stimname等部分标签  改用stim+split吧；而且它的csv文件是全的，tsv文件缺失信息
#注意sub-06感觉实验没做完就中断了；sub-49 sub-50用的127电极
# sub_st=1    #sub-01的tsv/csv标签文件未记录teststimnumber 直接取后2400个为测试集就行 


import math

import mne  
import numpy as np
import os
import pandas as pd


from utils_preprocess_22Australia import load_label_22Australia_train_data
from utils_preprocess_22Australia import load_label_22Australia_test_data
from utils_preprocess_22Australia import load_eeg_and_event_22Australia
from utils_preprocess_22Australia import epoch_22Australia
from utils_preprocess_22Australia import save_sample_and_label_22Australia
from utils_preprocess_22Australia import whether_already_saved


# 预处理参数
amplify_rate=1_000_000

downsample_freq = 1000
freq_low = 0.1
freq_high = 100
freq_notch = 50

# #作者的epoch长度
# epoch_st = -0.1  # 以s为单位
# epoch_end =  1 # 以s为单位
epoch_st = 0.0  # 以s为单位
epoch_end =  0.5 # 以s为单位

baseline_st = 0
baseline_end = 0

#一些记录信息
data_type='img_show'
paradigm_type='random_RSVP'
# 切分后存储位置
save_dir=''
save_dir=save_dir.replace('.','p')
save_dir=save_dir.replace(' ','_')

dataset_path=''


#sub 1 6 采集存在问题，作者和我都未处理
sub_list=[    2,3,4,5,
              7,8,9,10,
            11,12,13,14,15,16,17,18,19,20,
            21,22,23,24,25,26,27,28,29,30,
            31,32,33,34,35,36,37,38,39,40,
            41,42,43,44,45,46,47,48
]
for subNO in sub_list:
    #是否已处理过  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    whether_is_preprocessed=True
    for dataset in ['train','test']:
        if dataset=='train':
            #训练集
            concept_num=1854
            img_num_per_concept=12
            sample_num_per_sub=concept_num*img_num_per_concept
        elif dataset=='test':
            #测试集
            concept_num=200
            img_num_per_concept=1
            repeat_num=12
            sample_num_per_sub=concept_num*img_num_per_concept*repeat_num
        sub_dir='sub-'+str(subNO).zfill(2)
        sub_save_path=os.path.join(save_dir,sub_dir)  
        sub_save_path=os.path.join(sub_save_path,dataset) #我打算把train test的数据分开放
        #其实sub-06的原始数据应该是中间断了，只有12360个样本，暂时不处理他的了
        if whether_already_saved(sub_save_path,sample_num_per_sub):
            print(sub_dir,'already_saved!')
            continue
        else:
            whether_is_preprocessed=False #只要 训练集 或 测试集 中有1个未预处理过
            break

    #如果训练集+测试集均已经预处理过，就跳过该被试
    if whether_is_preprocessed:
        continue
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 



    # 读取数据
    raw_data, events=load_eeg_and_event_22Australia(dataset_path, subNO,
                       downsample_freq,
                       freq_low, freq_high,
                       freq_notch,)

    for dataset in ['train','test']:
        if dataset=='train':
            #训练集
            sub_dir='sub-'+str(subNO).zfill(2)
            sub_save_path=os.path.join(save_dir,sub_dir)  
            sub_save_path=os.path.join(sub_save_path,dataset) #我打算把train test的数据分开放
            # 读取并转化标签
            sample_label_0_1853,sample_label_onehot,sample_img_str,is_train_data=load_label_22Australia_train_data(dataset_path, subNO)
        elif dataset=='test':
            #测试集
            sub_dir='sub-'+str(subNO).zfill(2)
            sub_save_path=os.path.join(save_dir,sub_dir)  
            sub_save_path=os.path.join(sub_save_path,dataset) #我打算把train test的数据分开放
            # 读取并转化标签
            sample_label_0_1853,sample_label_onehot,sample_img_str,is_train_data=load_label_22Australia_test_data(dataset_path, subNO)



        #切分数据
        event_id_epoch=[10001]  # 只挑出图片展示的时刻
        sample_time_electrode=epoch_22Australia( raw_data,
        events,event_id_epoch,
                downsample_freq,
                epoch_st, epoch_end,
                baseline_st, baseline_end,)

                
        if dataset=='train':
            #训练集
            concept_num=1854
            img_num_per_concept=12
            sample_num_per_sub=concept_num*img_num_per_concept
            #由于我只保存了训练数据集的标签，因此这里把测试集的脑电数据删去
            sample_time_electrode=sample_time_electrode[is_train_data,]
        elif dataset=='test':
            #测试集
            concept_num=200
            img_num_per_concept=1
            repeat_num=12
            sample_num_per_sub=concept_num*img_num_per_concept*repeat_num
            sample_time_electrode=sample_time_electrode[~is_train_data,]
            #is_train_data其实就是最后2400张为测试集，实验时不告诉被试哪些是测试集

        #其实sub-06的应该是中间断了，只有12360个样本，暂时不处理他的了
        if sample_time_electrode.shape[0]!=sample_num_per_sub:
            print(sub_dir,' has bad epoches!')
            print('bad epoches 被自动去除了。这样标签不对应，故不保存该数据的样本')
            continue
    # ----------------------------------------------------------------------------------------------------
        # 保存
        overwrite_flag=0         
        save_sample_and_label_22Australia(sub_save_path, 
                                    sub_dir,
                            sample_time_electrode, sample_label_0_1853,sample_label_onehot,sample_img_str,
                            downsample_freq,
                            freq_low, freq_high,
                            freq_notch,
                            epoch_st, epoch_end,
                            baseline_st, baseline_end,
                            data_type,
                            paradigm_type,
                            overwrite_flag,
                            amplify_rate)
    # ----------------------------------------------------------------------------------------------------
