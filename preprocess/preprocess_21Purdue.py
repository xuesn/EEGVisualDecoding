# 基线校正是分电极的，并不一定有益


import mne  
import numpy as np
import os
import pandas as pd
import json
import math


# 预处理参数
amplify_rate=1_000_000
# 频域
downsample_freq = 250  # 原来为4096hz 
freq_low = 0.1
freq_high = 100
freq_notch = 50
# 切分长度 放映2s 休息1s
epoch_st = -0.1  # 以s为单位
epoch_end =  2.0 # 以s为单位
# 基线校正
baseline_st = -0.1
baseline_end = 0


# ----------------------------------------------------------------------------------------------------
#一些记录信息
data_type='img_show'
paradigm_type='random_slow'
# 切分后存储位置
save_dir='' 
save_dir=save_dir.replace('.','p')
save_dir=save_dir.replace(' ','_')
# ----------------------------------------------------------------------------------------------------
# 原始脑电数据和每组使用的图片文件名
data_path = '/CVPR2021-02785/data/'
label_path = '/CVPR2021-02785/design/'
# 数据文件列表
file_list = os.listdir(data_path)
temp_list = [fn for fn in file_list if fn.startswith("imagenet40-1000-1-")]  # bdf_filename = 'imagenet40-1000-1-00.bdf'
bdf_list = [fn for fn in temp_list if fn.endswith(".bdf")]
bdf_list.sort()
# 标签文件列表
file_list = os.listdir(label_path)
temp_list = [fn for fn in file_list if fn.startswith("run-")]  # txt_filename = 'run-00.txt'
txt_list = [fn for fn in temp_list if fn.endswith(".txt")]
txt_list.sort()
# 该两类文件sort后是一一对应的，但最好检查一下
# ----------------------------------------------------------------------------------------------------
finished_bdf_num = 0
for _, bdf_filename in enumerate(bdf_list[finished_bdf_num:]):
    # break
    bdfNO = int(bdf_filename.split('imagenet40-1000-1-')[-1].split('.bdf')[0])
    expNO_str = str(bdfNO).zfill(2)
    txt_filename = 'run-' + expNO_str + '.txt'
    # 检查一下data与label文件是否对应
    assert bdf_filename[-6:-4] == txt_filename[-6:-4], '！！！样本数据与标签不对应！！！'
# ----------------------------------------------------------------------------------------------------
    #检查是否已经预处理过
    save_path = os.path.join(save_dir,'exp'+expNO_str)
    if os.path.exists(save_path):
        file_list = os.listdir(save_path)
        if (len(file_list) == 800) and ('sample00399.npy' in file_list) and ('sample00399.json' in file_list):
            print(bdf_filename,'already_saved!')
            continue
# ----------------------------------------------------------------------------------------------------
    # 读取并转化标签
    txt_label = pd.read_table(
        label_path+txt_filename, header=None)
    str_label = txt_label.values
    sample_img_str = [str_[0] for str_ in str_label]

    # imagenet的图片命名规则，nxxxxxxxx代表类别，_xxxxx代表该类别编号
    # （编号位数不补零，即存在_x/_xx/_xxx/_xxxx，且编号不连续，应该是把低质量图片筛了一下？）
    classes = {"n02106662": 0,
               "n02124075": 1,
               "n02281787": 2,
               "n02389026": 3,
               "n02492035": 4,
               "n02504458": 5,
               "n02510455": 6,
               "n02607072": 7,
               "n02690373": 8,
               "n02906734": 9,
               "n02951358": 10,
               "n02992529": 11,
               "n03063599": 12,
               "n03100240": 13,
               "n03180011": 14,
               "n03272010": 15,
               "n03272562": 16,
               "n03297495": 17,
               "n03376595": 18,
               "n03445777": 19,
               "n03452741": 20,
               "n03584829": 21,
               "n03590841": 22,
               "n03709823": 23,
               "n03773504": 24,
               "n03775071": 25,
               "n03792782": 26,
               "n03792972": 27,
               "n03877472": 28,
               "n03888257": 29,
               "n03982430": 30,
               "n04044716": 31,
               "n04069434": 32,
               "n04086273": 33,
               "n04120489": 34,
               "n04555897": 35,
               "n07753592": 36,
               "n07873807": 37,
               "n11939491": 38,
               "n13054560": 39}
    # ！！！标签好提，直接load_txt，然后取每一行的前9个字符，对应地改为类别序号即可

    sample_num = len(txt_label)
    sample_label_0_39 = np.zeros([sample_num, ])  
    for i, str_ in enumerate(str_label):
        class_str = str_[0].split('_')[0]
        class_int = classes[class_str]
        sample_label_0_39[i] = class_int

    class_num = max(classes.values())+1  # 即40
    sample_label_onehot = np.zeros(
        [sample_label_0_39.shape[0], class_num])
    sample_label_0_39 = sample_label_0_39.astype(int)
    sample_label_onehot[np.arange(sample_label_0_39.size), sample_label_0_39] = 1

    # 可以修改0~39内的数，测试一下，是不是都是10张
    # sum(sample_label_0_39==36)
    # sum(sample_label_onehot[:,19]==1)
# ----------------------------------------------------------------------------------------------------
    # 读取数据
    raw_data = mne.io.read_raw_bdf(data_path+bdf_filename)
    # 每组实验20多min 2s放映 1s休息 一组40类各10张
    # 104 EEG, 1 Stimulus
    # raw_data.plot()
    # 电极
    # ch_names = raw_data.ch_names
    # A1~A32 B1~B32 C1~C32 EXG1~EXG8 Status
    # two occipital channels (C31 and C32)
    # 96 channels 也就是说我们用ABC即可
    # EXG 仪器图中有EX1-EX8，为额外备用的电极
    # plot信号中EXG3-8没信号，EXG1和2为左右耳垂，可作为参考电极
    # Status 图片展示时刻会有脉冲
# ----------------------------------------------------------------------------------------------------
    # 信号处理
    raw_data.load_data() 
    # 重参考
    raw_data.set_eeg_reference(ref_channels=['EXG1', 'EXG2'])  #11s
    # Two additional channels were used to record the signal from the earlobes for rereferencing.
    # 重设采样率 
    raw_data.resample(sfreq=downsample_freq)  # 1min34s
    # 滤波
    raw_data.filter(l_freq=freq_low, h_freq=freq_high,
                    fir_design='firwin')  # 250Hz时3s
    # notch 去单频率
    raw_data.notch_filter(
        np.arange(freq_notch, freq_high, freq_notch))  # 50为直流电频 100为其倍频
    # (freq_notch, freq_high, freq_notch)意思是notch掉freq_notch，然后从freq_notch处开始步长取freq_notch，直到freq_high
# ----------------------------------------------------------------------------------------------------
    # onset，图片展示时刻
    # events, event_id = mne.events_from_annotations(raw_data) # 不适用于21Purdue
    # drop前才行
    events = mne.find_events(raw_data, min_duration=1/downsample_freq,shortest_event=0,initial_event=True)
    event_id_epoch = [65281]  # 只挑出图片展示的时刻  131071 131069在开始时会出现
# ----------------------------------------------------------------------------------------------------
    # 未连接和不使用的电极
    raw_data.drop_channels(
        ch_names=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'])  
    electrode_num = 96
# ----------------------------------------------------------------------------------------------------
    # 切分
    epochs = mne.Epochs(raw_data, events, event_id_epoch, tmin=epoch_st, tmax=epoch_end,
                        baseline=(baseline_st, baseline_end), picks=None, preload=False,
                        reject=None, flat=None, proj=True, decim=1,
                        reject_tmin=None, reject_tmax=None, detrend=None,
                        on_missing='raise', reject_by_annotation=True, metadata=None,
                        event_repeated='error', verbose=None)
    # 转np数组
    sample_time_electrode = epochs.get_data().transpose(0, 2, 1)
    sample_time_electrode = sample_time_electrode[:, :-1, :]
    # 若cpu内存不足，就可以del
    # del epochs
    print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
    print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
    print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
    print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
    print('sample_time_electrode-shape:', sample_time_electrode.shape)
# ----------------------------------------------------------------------------------------------------
    # 保存
    save_path = os.path.join(save_dir,'exp'+expNO_str)
    sample_info_rear = 'exp'+expNO_str+ ' ' +\
        str(downsample_freq)+'Hz' + ' ' +\
        'bp' + str(freq_low)+'_' + str(freq_high)+'Hz' + ' ' +\
        'notch' + str(freq_notch)+'Hz' + ' ' +\
        str(int(epoch_st*1000))+'_' + str(int(epoch_end*1000))+'ms' + ' ' +\
        'bc' + str(int(baseline_st*1000))+'_' + str(int(baseline_end*1000))+'ms' + ' ' +\
        data_type+' '+paradigm_type+'.json'
    # 逐样本保存
    sample_num = sample_time_electrode.shape[0]
    if sample_num != sample_label_0_39.shape[0]:
        assert False,'数据的样本数与标签的样本数不相等'
    for sampleNO in range(sample_num):
        # 每个文件夹分5000个存储
        max_sample_num_one_path=5000
        sampleNO_st=math.floor(sampleNO/max_sample_num_one_path)*max_sample_num_one_path
        sampleNO_end=sampleNO_st+max_sample_num_one_path-1
        divide_dir='sample'+str(sampleNO_st).zfill(5)+'_'+str(sampleNO_end).zfill(5)
        if sample_num > max_sample_num_one_path: #当样本数不多时，没必要分5000存储
            divide_save_path=os.path.join(save_path,divide_dir)
        else:
            divide_save_path=save_path
        # print(divide_save_path)
        # 创建文件夹
        if not os.path.exists(divide_save_path):
            os.makedirs(divide_save_path)
            print(divide_save_path, ' created!')
        
        sampleNO_str = 'sample'+str(sampleNO).zfill(5)  # 补零到5个
        # 词典
        label = sample_label_0_39[sampleNO]
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
        overwrite_flag = 0
        if (not os.path.exists(json_full_path)) or (overwrite_flag == 1): 
            with open(json_full_path, "w") as f:
                json.dump(sample_json_dict, f, indent=2)
            if sampleNO%400==399:
                print('!!!'+json_full_path+' saved!!!')
        #存脑电数据为npy，json可能是按字符存的，空间要大7倍
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





