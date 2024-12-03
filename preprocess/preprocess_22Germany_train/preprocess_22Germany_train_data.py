





import mne  
import numpy as np
import os
import pandas as pd


from utils_preprocess_22Germany import load_label_22Germany_train_data
from utils_preprocess_22Germany import load_eeg_and_event_22Germany_train_data
from utils_preprocess_22Germany import epoch_22Germany
from utils_preprocess_22Germany import save_sample_and_label_22Germany
from utils_preprocess_22Germany import whether_already_saved

# 预处理参数
amplify_rate=1_000_000

downsample_freq = 1000
freq_low = 0.1
freq_high = 100
freq_notch = 50

# #澳洲作者的epoch长度  德国的作者是-0.2~0.8
# epoch_st = -0.1  # 以s为单位
# epoch_end =  1 # 以s为单位
epoch_st = 0.0  # 以s为单位
epoch_end =  0.5 # 以s为单位

baseline_st = 0
baseline_end = 0


# 每组实验1.5h 1s放映5张，且我只预处理了训练集

#一些记录信息
data_type='img_show'
paradigm_type='random_RSVP'

# 切分后存储位置
save_dir=''
save_dir=save_dir.replace('.','p')
save_dir=save_dir.replace(' ','_')



raw_data_path='/share/eeg_datasets/Vision/22_Germany/Raw-EEG-data/'
img_path='/share/eeg_datasets/Vision/22_Germany/Image-set/'
label_file=os.path.join(img_path,'image_metadata.npy')


sub_num=10
session_num=4
for subNO in range(1,sub_num+1):
    sub_dir='sub-'+str(subNO).zfill(2)
    sub_save_path=os.path.join(save_dir,sub_dir)  
    for sessionNO in range(1,session_num+1):
        session_dir='ses-'+str(sessionNO).zfill(2)
        session_save_path=os.path.join(sub_save_path,session_dir)
        session_save_path=os.path.join(session_save_path,'train') #我打算把train test的数据分开放，因此该代码把数据存在train文件夹中   
        #是否已处理过
        concept_num=1654
        img_num_per_concept=10
        sample_num_per_sub=concept_num*img_num_per_concept
        #实际上每次都多放映了一些图片
        # 每个sess都是16710张图像 90张巴斯光年99999
        # 比16540多出来的是什么？
        # 16540小类中
        # 出现4次的有：15866个
        # 出现5次的有：668个
        # 出现6次的有：6个
        # #作者每次直接去掉了多的:
        # # Select only a maximum number of EEG repetitions
        # if data_part == 'test':
        #     max_rep = 20
        # else:
        #     max_rep = 2
        sample_num_per_sub=16710
        if whether_already_saved(session_save_path,sample_num_per_sub):
            print(sub_dir,session_dir,'already_saved!')
            continue


        # break

        # 读取数据
        raw_data, events=load_eeg_and_event_22Germany_train_data(raw_data_path, 
                        subNO,sessionNO,
                       downsample_freq,
                       freq_low, freq_high,
                       freq_notch)
        # 读取并转化标签
        sample_label_0_1853,sample_label_onehot,sample_img_str=load_label_22Germany_train_data(img_path, events)
        #切分数据
        event_id=set(events[:,2])
        special_img_id=99999#巴斯光年的id号
        event_id.remove(special_img_id) # 只挑出正常图片展示的时刻，不要巴斯光年
        event_id_epoch=list(event_id)
        sample_time_electrode=epoch_22Germany( raw_data,
        events,event_id_epoch,
                downsample_freq,
                epoch_st, epoch_end,
                baseline_st, baseline_end,)

        
        if sample_time_electrode.shape[0]!=sample_num_per_sub:
            print(sub_dir,' has bad epoches!')
            print('bad epoches 被自动去除了。这样标签不对应，故不保存该数据的样本')
            continue
    # ----------------------------------------------------------------------------------------------------
        # 保存
        overwrite_flag=0         
        save_sample_and_label_22Germany(session_save_path, 
                                    sub_dir,session_dir,
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








