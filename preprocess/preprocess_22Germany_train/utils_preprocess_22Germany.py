
import mne
import numpy as np
import os

import json

import operator

import math





# 训练集：1654 concept 各放映10张 重复放映4次
# 测试集：取 其他的200 concept 各取1张 重复放映 80次
#测试集暂未处理，先预处理训练集的

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
def load_label_22Germany_train_data(img_path, events):
    imgNO=(events[:,2])
    special_img_id=99999#巴斯光年的id号
    is_special=imgNO==special_img_id
    imgNO_THINGS=imgNO[~is_special]

    # 获取每个样本的1654小类标签
    # imgNO_THINGS为16540图片序号标签，÷10就是1654小类
    # sample_label_1654是1654中的序号；而classNO_in_THINGS是1854中的序号
    # train_img_concepts_THINGS = sample_label_1654  # 1~1654

    meta_file = 'image_metadata.npy'
    meta_full_path = os.path.join(img_path, meta_file)
    meta_matrix = np.load(meta_full_path, allow_pickle=True)
    meta_dict = meta_matrix.item()
    meta_dict.keys()
    #dict_keys(['test_img_concepts', 'test_img_concepts_THINGS', 'test_img_files', 
    # 'train_img_files', 'train_img_concepts', 'train_img_concepts_THINGS'])
    train_img_concepts_THINGS = meta_dict['train_img_concepts_THINGS']
    train_img_files = meta_dict['train_img_files']
    train_img_concepts = meta_dict['train_img_concepts']

    sample_img_str=np.array(train_img_files)[(imgNO_THINGS-1)]
    sample_img_concepts_THINGS=np.array(train_img_concepts_THINGS)[(imgNO_THINGS-1)]
    sample_label_0_1853=np.array([ int(str.split('_')[0])-1  for str in sample_img_concepts_THINGS])

    class_num = max([ int(str.split('_')[0])  for str in train_img_concepts_THINGS])   # 即1854
    sample_label_onehot = np.zeros(
        [sample_label_0_1853.shape[0], class_num])
    sample_label_onehot[np.arange(sample_label_0_1853.size), sample_label_0_1853] = 1

    #返回值为Series类型，当成array用吧
    return sample_label_0_1853,sample_label_onehot,sample_img_str


# 读取原始数据并做基本预处理+读取event、event_id_epoch----------------------------------------------------------------------------------------------------
subNO=1
sessionNO=1
raw_data_path='/share/eeg_datasets/Vision/22_Germany/Raw EEG data/'

def load_eeg_and_event_22Germany_train_data(raw_data_path,
                         subNO,sessionNO,
                       downsample_freq,
                       freq_low, freq_high,
                       freq_notch,):
    sub_dir='sub-'+str(subNO).zfill(2)
    session_dir='ses-'+str(sessionNO).zfill(2)
    
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

    train_concept_num = 1654
    imgnum_per_concept = 10

    file_path = os.path.join(raw_data_path, sub_dir, session_dir)

    #其实不是最原始的数据，而是带通滤波过的数据
    train_file_str = 'raw_eeg_training.npy'
    test_file_str = 'raw_eeg_test.npy'

    train_file_path = os.path.join(file_path, train_file_str)
    test_file_path = os.path.join(file_path, test_file_str)

    print('Ready load', train_file_path)
    train_matrix = np.load(
        train_file_path, allow_pickle=True)  # 2.6g 148得个16s装载进来 52得个666min也没装载进来、只要不卡，用不了多少秒
    train_dict = train_matrix.item()
    train_dict.keys()
    # dict_keys(['raw_eeg_data', 'ch_names', 'ch_types',
    #            'sfreq', 'highpass', 'lowpass'])
    # 电极、相对图片展示时的采样时刻
    ch_names=train_dict['ch_names']  # 63+'stim'
    train_dict['ch_types']  # 63-eeg+1-stim
    ori_sfreq = train_dict['sfreq']  # 1000hz
    #这里指的是滤波器为高通还是低通，而不是频带范围
    highpass=train_dict['highpass']  # 0.01hz
    lowpass=train_dict['lowpass']  # 100hz

    # eeg各维度：对应图片*重复放映*电极*采样时刻
    train_eeg = train_dict['raw_eeg_data']
    train_eeg.shape  # (64, 5450560)

    total_electrode_num = 63
    data=train_eeg[:total_electrode_num, ]
    stim = train_eeg[total_electrode_num, ]
    # print(np.sum(stim == 0))  # 5433760
    # print(np.sum(stim > 0))  # 16800：16710张图像 90张巴斯光年
    # print(np.sum(stim == 1))  # 2
    # print(np.sum(stim == 2))  # 2
    # print(np.sum(stim == 16539))  # 0
    # print(np.sum(stim == 16540))  # 0
    # print(np.sum(stim == 16800))  # 0
    # print(np.sum(stim == 16799))  # 0
    # print(np.sum(stim == 99999))  # 巴斯光年99999
    # 每个训练集session。是取的16540张图片中的 约一半 放映的，每一张图片重复放映了2次

    ch_types=[ 'eeg']*63+['stim']
    info=mne.create_info(ch_names, sfreq=ori_sfreq, ch_types=ch_types, verbose=None)
    raw_data=mne.io.RawArray(train_eeg, info, first_samp=0, copy='auto', verbose=None)

    raw_data.load_data()
    # 63电极+1 stim
    # raw_data.drop_channels(ch_names=['stim', ]) 不用 读取时可能自动作为event了
    # 滤波
    raw_data.filter(l_freq=freq_low, h_freq=freq_high,
                    fir_design='firwin')  
    # notch 去单频率
    if freq_notch is not None:
        raw_data.notch_filter(
            np.arange(freq_notch, freq_high, freq_notch), verbose='warning')  # 50为交流电频 100为其倍频
    # (freq_notch, freq_high, freq_notch)意思是notch掉freq_notch，然后从freq_notch处开始步长取freq_notch，直到freq_high
    # 重设采样率
    raw_data.resample(sfreq=downsample_freq)

    events = mne.find_events(raw_data, min_duration=1/downsample_freq,shortest_event=0,initial_event=True)
    raw_data.drop_channels(ch_names=['stim']) 
    return raw_data, events


# 切分方案--每组实验20多min 2s放映 1s休息 一组40类各10张----------------------------------------------------------------------------------------------------
# event_id_epoch = [10001]  # 只挑出图片展示的时刻
def epoch_22Germany( raw_data,
               events,event_id_epoch,
               downsample_freq,
               epoch_st, epoch_end,
               baseline_st, baseline_end,): 
    if baseline_st==baseline_end:#代表不需要做baseline-correction
        baseline=None
    else:
        baseline=(baseline_st, baseline_end)
    # duration_sec = epoch_end-epoch_st
    # timepoint_num = int(duration_sec*downsample_freq)
    epochs = mne.Epochs(raw_data, events, event_id_epoch, tmin=epoch_st, tmax=epoch_end,
                        baseline=baseline, picks=None, preload=False,
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
def save_sample_and_label_22Germany(save_path, 
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
                          amplify_rate):  # --每1张图片保存为1个样本  
    # 样本信息
    sample_info_rear = sub_dir+ ' ' +\
       session_dir+ ' ' +\
        str(downsample_freq)+'Hz' + ' ' +\
        'bp' + str(freq_low)+'_' + str(freq_high)+'Hz' + ' ' +\
        'notch' + str(freq_notch)+'Hz' + ' ' +\
        str(int(epoch_st*1000))+'_' + str(int(epoch_end*1000))+'ms' + ' ' +\
        'bc' + str(int(baseline_st*1000))+'_' + str(int(baseline_end*1000))+'ms' + ' ' +\
        data_type+' '+paradigm_type+'.json'
    # 逐样本保存
    sample_num = sample_time_electrode.shape[0]
    if sample_num != sample_label_0_1853.shape[0]:
        assert False,'数据的样本数与标签的样本数不相等'
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
        label = sample_label_0_1853[sampleNO, ]
        label_onehot = sample_label_onehot[sampleNO, ]
        img_str=sample_img_str[sampleNO]
        sample_info = sampleNO_str+' '+sample_info_rear
        sample_info = sample_info.replace(' ', '_')
        sample_json_dict = {
            "label": label.tolist(),
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



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#转27大类时用

import pandas as pd
'''
label_1854=[1,2,3]
label_0_1853=range(1854)
'''
# 1854个小类对应的27大类
def things_1854to27(label_0_1853):
    # 我决定取category_mat_bottom_up，category_mat_manual，category_mat_top_down的并。
    higher_category_path = '/27 higher-level categories/'
    label_bu_str = 'category_mat_bottom_up'
    label_m_str = 'category_mat_manual'
    label_td_str = 'category_mat_top_down'
    # 读tsv
    #bottom_up
    label_bu_df = pd.read_csv(os.path.join(
        higher_category_path, label_bu_str+'.tsv'), sep='\t', header=0)
    label_bu = label_bu_df.values
    #manual
    label_m_df = pd.read_csv(os.path.join(
        higher_category_path, label_m_str+'.tsv'), sep='\t', header=0)
    label_m = label_m_df.values
    #top_down
    label_td_df = pd.read_csv(os.path.join(
        higher_category_path, label_td_str+'.tsv'), sep='\t', header=0)
    label_td = label_td_df.values
    # 取category_mat_bottom_up，category_mat_manual，category_mat_top_down的并
    label_3in1 = ((label_bu+label_m+label_td) > 0)  

    # 获取每个样本的27大类标签
    #由于存在多类别数据，所以用multihot的格式存储
    label_27_multihot = label_3in1[label_0_1853].astype('int')

    return label_27_multihot
