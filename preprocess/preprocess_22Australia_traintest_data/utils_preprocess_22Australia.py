
# 每组实验1h 1s放映10张
# 我只预处理了训练集：
# 训练集：1854 concept 各放映12张，不重复放映
#测试集：\code_and_figure\test_images.csv  200个concept各挑了1张，重复放映 12次
 


#  sub-06应该是中断了 只有1728.099 secs.
#注意sub-06感觉实验没做完就中断了；sub-49 sub-50用的127电极

from pandas import Series
import pandas as pd
import mne
import numpy as np
import os

import json

import operator

import math


from get_concept_0_1853_dict import get_concept_0_1853_dict



# 若已经预处理并保存过，则不再重复处理----------------------------------------------------------------------------------------------------
def whether_already_saved(sub_save_path,sample_num_per_sub):
    max_sample_num_one_path=5000
    final_sampleNO_st=math.floor((sample_num_per_sub-1)/max_sample_num_one_path)*max_sample_num_one_path
    final_sampleNO_end=final_sampleNO_st+max_sample_num_one_path-1
    final_divide_dir='sample'+str(final_sampleNO_st).zfill(5)+'_'+str(final_sampleNO_end).zfill(5)
    final_divide_save_path=os.path.join(sub_save_path,final_divide_dir)
    #只检查最后一个文件夹，其实有漏洞
    if os.path.exists(final_divide_save_path):
        current_file_list=os.listdir(final_divide_save_path)
        #构建完整样本列表
        full_sample_file_list=[]
        for sampleNO in range(final_sampleNO_st,sample_num_per_sub):
            sampleNO_str = 'sample'+str(sampleNO).zfill(5)  # 补零到5个
            full_sample_file_list.append(sampleNO_str+'.json' ) # 补零到5个
            full_sample_file_list.append(sampleNO_str+'.npy' ) # 补零到5个
        return operator.eq(current_file_list,full_sample_file_list)
    else:
        return False

        


# 读取标签----------------------------------------------------------------------------------------------------
def load_label_22Australia_train_data(dataset_path, subNO):
    # 小类标签0~1853
    # 数据集中有两个不同后缀的文件都叫rsvp_events，但tsv比csv多了前2列，即图片展示时刻和时长
    sub_dir='sub-'+str(subNO).zfill(2)

    #tsv文件应该是实验时逐行记录的，比如sub-06中途中断，tsv记录的就比csv少一半内容
    tsv_fname=sub_dir+'_task-rsvp_events.tsv'
    sub_rsvp_events_path = os.path.join(dataset_path,sub_dir,'eeg',tsv_fname)
    sub_rsvp_events = pd.read_csv(sub_rsvp_events_path, sep='\t')
    
    #csv文件应该是实验前就生成好的
    # csv_fname=sub_dir+'_task-rsvp_events.csv'
    # sub_rsvp_events_path = os.path.join(dataset_path,sub_dir,'eeg',csv_fname)
    # sub_rsvp_events = pd.read_csv(sub_rsvp_events_path, sep=',')

    objectnumber = sub_rsvp_events.objectnumber  # 训练集 0~1853小类 各12张；测试集置为-1
    #注意sub-01的tsv、csv文件特殊，不含有teststimnumber等部分标签，可能是作者一开始的代码没记录这些信息，从sub2开始记录
    # 注意sub-06的tsv、csv文件特殊，不含有stimname等部分标签  改用stim+split吧；而且它的csv文件是全的，tsv文件缺失信息

    # teststimnumber = sub_rsvp_events.teststimnumber  # 测试集 0~199图片 重复放映12次；训练集置为-1
    # stimname=sub_rsvp_events.stimname#图片名
    stimname_fpath=sub_rsvp_events.stim#图片名
    stimname=[fp.split('\\')[-1] for fp in stimname_fpath]
    stimname=Series(stimname)

    # 取objectnumber中不为-1的，作为训练集的小类标签0~1853
    # print(objectnumber[-200*12:])  # 后200*12张为测试集，标记为-1
    is_train_data = objectnumber > -1
    sample_label_0_1853 = objectnumber[is_train_data]
    sample_img_str = stimname[is_train_data]

    # time_stimon = sub_rsvp_events.time_stimon
    # # time_stimon可以用于切分数据
    # time_stimon  # 以秒为单位
    
    class_num = np.max(sample_label_0_1853)+1  # 即1854
    sample_label_onehot = np.zeros(
        [sample_label_0_1853.shape[0], class_num])
    sample_label_onehot[np.arange(sample_label_0_1853.size), sample_label_0_1853] = 1

    #返回值为Series类型，当成array用吧
    return list(sample_label_0_1853),sample_label_onehot,list(sample_img_str),is_train_data


def load_label_22Australia_test_data(dataset_path, subNO):
    # 小类标签0~1853
    # 数据集中有两个不同后缀的文件都叫rsvp_events，但tsv比csv多了前2列，即图片展示时刻和时长
    sub_dir='sub-'+str(subNO).zfill(2)

    #tsv文件应该是实验时逐行记录的，比如sub-06中途中断，tsv记录的就比csv少一半内容
    tsv_fname=sub_dir+'_task-rsvp_events.tsv'
    sub_rsvp_events_path = os.path.join(dataset_path,sub_dir,'eeg',tsv_fname)
    sub_rsvp_events = pd.read_csv(sub_rsvp_events_path, sep='\t')
    
    #csv文件应该是实验前就生成好的
    # csv_fname=sub_dir+'_task-rsvp_events.csv'
    # sub_rsvp_events_path = os.path.join(dataset_path,sub_dir,'eeg',csv_fname)
    # sub_rsvp_events = pd.read_csv(sub_rsvp_events_path, sep=',')

    objectnumber = sub_rsvp_events.objectnumber  # 训练集 0~1853小类 各12张；测试集置为-1
    #注意sub-01的tsv、csv文件特殊，不含有teststimnumber等部分标签，可能是作者一开始的代码没记录这些信息，从sub2开始记录
    # 注意sub-06的tsv、csv文件特殊，不含有stimname等部分标签  改用stim+split吧；而且它的csv文件是全的，tsv文件缺失信息

    # teststimnumber = sub_rsvp_events.teststimnumber  # 测试集 0~199图片 重复放映12次；训练集置为-1
    # stimname=sub_rsvp_events.stimname#图片名
    stimname_fpath=sub_rsvp_events.stim#图片名
    stimname=[fp.split('\\')[-1] for fp in stimname_fpath]
    stim_concept=[fp.split('\\')[-2] for fp in stimname_fpath]
    stimname=Series(stimname)
    stim_concept=Series(stim_concept)


    # 取objectnumber中不为-1的，作为训练集的小类标签0~1853
    # print(objectnumber[-200*12:])  # 后200*12张为测试集，标记为-1
    is_train_data = objectnumber > -1
    #训练集
    # sample_label_0_1853 = objectnumber[is_train_data]
    # sample_img_str = stimname[is_train_data]
    #测试集
    sample_img_str = stimname[~is_train_data]
    sample_stim_concept=stim_concept[~is_train_data]
    #小类号in1853 要根据str得到
    concept_0_1853_dict= get_concept_0_1853_dict()
    sample_label_0_1853 = [concept_0_1853_dict[fp] for fp in sample_stim_concept]
    # sample_label_0_1853=Series(sample_label_0_1853)

    # time_stimon = sub_rsvp_events.time_stimon
    # # time_stimon可以用于切分数据
    # time_stimon  # 以秒为单位
    
    # label转为one-hot格式，（不保存了，1854类还是不小的）
    class_num = np.max(sample_label_0_1853)+1  # 即1854
    sample_label_onehot = np.zeros(
        [len(sample_label_0_1853), class_num])
    sample_label_onehot[np.arange(len(sample_label_0_1853)), sample_label_0_1853] = 1

    #返回值为Series类型，当成array用吧
    return sample_label_0_1853,sample_label_onehot,list(sample_img_str),is_train_data


# 读取原始数据并做基本预处理+读取event、event_id_epoch----------------------------------------------------------------------------------------------------
def load_eeg_and_event_22Australia(dataset_path, subNO,
                       downsample_freq,
                       freq_low, freq_high,
                       freq_notch,):
    
    sub_dir='sub-'+str(subNO).zfill(2)
    vhdr_fname=sub_dir+'_task-rsvp_eeg.vhdr'
    vhdr_path = os.path.join(dataset_path,sub_dir,'eeg',vhdr_fname)

    # 读BrainVision ActiChamp设备记录的数据
    raw_data = mne.io.read_raw_brainvision(vhdr_path, preload=False)
    ch_names = raw_data.ch_names  # 64导少了Cz-可能被作为参考电极吧 # 128导少了FCz-可能被作为参考电极吧

    raw_data.load_data()
    # 63电极+1 stim
# sub01~48
#     {
# 	"TaskName":"rsvp",
# 	"PowerLineFrequency":50,
# 	"SamplingFrequency":1000,
# 	"EEGChannelCount":63,
# 	"EOGChannelCount":0,
# 	"ECGChannelCount":0,
# 	"EMGChannelCount":0,
# 	"EEGReference":"Cz",
# 	"SoftwareFilters":"n/a"
# }
    # 虽然这里写的128，但其实127 EEG +1 stim
# sub-49/50
# {
# 	"TaskName":"rsvp",
# 	"PowerLineFrequency":50,
# 	"SamplingFrequency":1000,
# 	"EEGChannelCount":128,
# 	"EOGChannelCount":0,
# 	"ECGChannelCount":0,
# 	"EMGChannelCount":0,
# 	"EEGReference":"FCz",
# 	"SoftwareFilters":"n/a"
# }


    # raw_data.drop_channels(ch_names=['stim', ]) 不用 读取时可能自动作为event了
    # 滤波（滤波和时间平滑要在下采样之前做）
    raw_data.filter(l_freq=freq_low, h_freq=freq_high,
                    fir_design='firwin')  
    # notch 去单频率
    raw_data.notch_filter(
        np.arange(freq_notch, freq_high, freq_notch))  # 50为交流电频 100为其倍频
    # (freq_notch, freq_high, freq_notch)意思是notch掉freq_notch，然后从freq_notch处开始步长取freq_notch，直到freq_high
    # 重设采样率
    raw_data.resample(sfreq=downsample_freq)

    events, event_id = mne.events_from_annotations(raw_data)
    return raw_data, events



# 切分方案--每组实验20多min 2s放映 1s休息 一组40类各10张----------------------------------------------------------------------------------------------------
# event_id_epoch = [10001]  # 只挑出图片展示的时刻
def epoch_22Australia( raw_data,
               events,event_id_epoch,
               downsample_freq,
               epoch_st, epoch_end,
               baseline_st, baseline_end,): 
    # duration_sec = epoch_end-epoch_st
    # timepoint_num = int(duration_sec*downsample_freq)
    epochs = mne.Epochs(raw_data, events, event_id_epoch, tmin=epoch_st, tmax=epoch_end,
                        baseline=(baseline_st, baseline_end), picks=None, preload=False,
                        reject=None, flat=None, proj=True, decim=1,
                        reject_tmin=None, reject_tmax=None, detrend=None,
                        on_missing='raise', reject_by_annotation=True, metadata=None,
                        event_repeated='error', verbose=None)
    # 转np数组
    sample_time_electrode = epochs.get_data().transpose(0, 2, 1)
    # mne.Epochs会多切1个值，因此去除
    sample_time_electrode = sample_time_electrode[:, :-1, :]
    # 若cpu内存不足，就可以del
    # del epochs
    print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    print('sample_time_electrode-shape:', sample_time_electrode.shape)
    return sample_time_electrode


# 保存----------------------------------------------------------------------------------------------------
def save_sample_and_label_22Australia(save_path, 
                                sub_name,
                          sample_time_electrode, sample_label_0_1853,sample_label_onehot,sample_img_str,
                          downsample_freq,
                          freq_low, freq_high,
                          freq_notch,
                          epoch_st, epoch_end,
                          baseline_st, baseline_end,
                          data_type,
                          paradigm_type,
                          overwrite_flag,
                          amplify_rate):  # --每1张图片保存为1个样本  
    # 样本信息
    sample_info_rear = sub_name+ ' ' +\
        str(downsample_freq)+'Hz' + ' ' +\
        'bp' + str(freq_low)+'_' + str(freq_high)+'Hz' + ' ' +\
        'notch' + str(freq_notch)+'Hz' + ' ' +\
        str(int(epoch_st*1000))+'_' + str(int(epoch_end*1000))+'ms' + ' ' +\
        'bc' + str(int(baseline_st*1000))+'_' + str(int(baseline_end*1000))+'ms' + ' ' +\
        data_type+' '+paradigm_type+'.json'
    # 逐样本保存
    sample_num = sample_time_electrode.shape[0]
    if sample_num != len(sample_label_0_1853):
        assert False, '数据的样本数与标签的样本数不相等'
    for sampleNO in range(sample_num):
        # 每个文件夹分5000个存储

        max_sample_num_one_path=5000
        sampleNO_st=math.floor(sampleNO/max_sample_num_one_path)*max_sample_num_one_path
        sampleNO_end=sampleNO_st+max_sample_num_one_path-1
        divide_dir='sample'+str(sampleNO_st).zfill(5)+'_'+str(sampleNO_end).zfill(5)
        divide_save_path=os.path.join(save_path,divide_dir)

        # print(divide_save_path)


        # 创建文件夹
        if not os.path.exists(divide_save_path):
            os.makedirs(divide_save_path)
            print(divide_save_path, ' created!')


        sampleNO_str = 'sample'+str(sampleNO).zfill(5)  # 补零到5个
        # 词典
        label = sample_label_0_1853[sampleNO]
        label_onehot = sample_label_onehot[sampleNO, ]
        img_str=sample_img_str[sampleNO]
        sample_info = sampleNO_str+' '+sample_info_rear
        sample_info = sample_info.replace(' ', '_')
        sample_json_dict = {
            # "label": label.tolist(),
            "label": label,
            # "label_onehot": label_onehot.tolist(),
            "image": img_str,
            "info": sample_info,
        }
        #存标签等信息
        json_fname = sampleNO_str+'.json'
        json_full_path = os.path.join(divide_save_path, json_fname)
        # 如果文件尚未被创建 或者 overwrite_flag==1，则写入
        if (not os.path.exists(json_full_path)) or (overwrite_flag == 1): 
            with open(json_full_path, "w") as f:
                json.dump(sample_json_dict, f, indent=2)
            if sampleNO%1000==999:
                print('!!!'+json_full_path+' saved!!!')
        #存脑电数据为npy
        eeg = sample_time_electrode[sampleNO, :, :]*amplify_rate#乘上amplify_rate，避免精度损失
        npy_fname = sampleNO_str+'.npy'
        npy_full_path = os.path.join(divide_save_path, npy_fname)
        # 如果文件尚未被创建 或者 overwrite_flag==1，则写入
        if (not os.path.exists(npy_full_path)) or (overwrite_flag == 1): 
            # np.save(npy_full_path,eeg)
            #存为float32，应该能节省一半的空间
            np.save(npy_full_path,eeg.astype(np.float32))
            if sampleNO%1000==999:
                print('!!!'+npy_full_path+' saved!!!')
    return
