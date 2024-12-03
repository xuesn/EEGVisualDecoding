


import os
import math
import json
import numpy as np
import scipy.io as scio


overwrite_flag = 1
# load
author_data_dir=''
sub_num = 10
for subNO in range(1,sub_num+1):
    fname = 'S'+str(subNO)+'.mat'
    fpath=os.path.join(author_data_dir,fname)
    data_dict = scio.loadmat(fpath)
    # break

    data_dict.keys()
    # dict_keys(['__header__', '__version__', '__globals__', 
    #           'sub', 'Fs', 'N', 'T', 
    #           'exemplarLabels', 'categoryLabels', 'X_2D', 'X_3D'])

    sub = data_dict['sub']  # S1
    Fs = data_dict['Fs']  # 62.5Hz
    N = data_dict['N']  # 每张图片32个时间点 对应62.5Hz的496ms
    T = data_dict['T']  # 5188个样本 72张*72次重复放映
    exemplarLabels = data_dict['exemplarLabels']  # 72张图片 1*5188
    categoryLabels = data_dict['categoryLabels']  # 6大类 1*5188
    X_2D = data_dict['X_2D']  # 5188*3968
    X_3D = data_dict['X_3D']  # 电极124*时间点32*样本数5188
    # --------------------------------------------------------
    print('mean:',np.mean(X_2D))
    print('max:',np.max(X_2D))
    print('min:',np.min(X_2D))
    print('std:',np.std(X_2D))
    # --------------------------------------------------------
    #一些记录信息
    data_type='img_show'
    paradigm_type='random_slow'
    # 切分后存储位置
    downsample_freq=Fs[0][0]
    epoch_st=0
    epoch_end=0.496
    baseline_st=0
    baseline_end=0
    freq_low=1
    freq_high=25
    save_dir=''
    save_dir=save_dir.replace('.','p')
    save_dir=save_dir.replace(' ','_')
    # --------------------------------------------------------
    # 保存
    save_path = os.path.join(save_dir, 'S'+str(subNO))
    # 样本信息  0810存的忘了加'exp'了，虽然无所谓啦
    freq_notch = ''
    sample_info_rear = 'S'+str(subNO)+ ' ' +\
        str(downsample_freq)+'Hz' + ' ' +\
        'bp' + str(freq_low)+'_' + str(freq_high)+'Hz' + ' ' +\
        'notch' + str(freq_notch)+'Hz' + ' ' +\
        str(int(epoch_st*1000))+'_' + str(int(epoch_end*1000))+'ms' + ' ' +\
        'bc' + str(int(baseline_st*1000))+'_' + str(int(baseline_end*1000))+'ms' + ' ' +\
        data_type+' '+paradigm_type+'.json'
    # 逐样本保存
    # X_3D   # 电极124*时间点32*样本数5188
    exemplarLabels = data_dict['exemplarLabels']  # 72张图片 1*5188
    categoryLabels = data_dict['categoryLabels']  # 6大类 1*5188
    sample_time_electrode=X_3D.transpose(2,1,0)
    sample_num = sample_time_electrode.shape[0]
    sample_label_0_5=categoryLabels.reshape(sample_num,)-1  
    sample_img_str=exemplarLabels.reshape(sample_num,)  
    # label转为one-hot格式
    class_num = max(sample_label_0_5)+1  # 即6
    print('class_num:',class_num)
    sample_label_onehot = np.zeros(
        [sample_label_0_5.shape[0], class_num])
    sample_label_0_5 = sample_label_0_5.astype(int)
    sample_label_onehot[np.arange(sample_label_0_5.size), sample_label_0_5] = 1

    for sampleNO in range(sample_num):
        # 每个文件夹分5188个存储
        max_sample_num_one_path=5188
        sampleNO_st=math.floor(sampleNO/max_sample_num_one_path)*max_sample_num_one_path
        sampleNO_end=sampleNO_st+max_sample_num_one_path-1
        divide_dir='sample'+str(sampleNO_st).zfill(5)+'_'+str(sampleNO_end).zfill(5)
        if sample_num > max_sample_num_one_path: #当样本数不多时，没必要分5188存储
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
        label = sample_label_0_5[sampleNO]
        label_onehot = sample_label_onehot[sampleNO, ]
        img_str=sample_img_str[sampleNO]
        sample_info = sampleNO_str+' '+sample_info_rear
        sample_info = sample_info.replace(' ', '_')
        
        img_str=img_str.tolist()
        sample_json_dict = {
            "label": label.tolist(),
            "image": img_str,
            "info": sample_info,
        }
        #存标签等信息
        json_fname = sampleNO_str+'.json'
        json_full_path = os.path.join(divide_save_path, json_fname)
        # 如果文件尚未被创建 或者 overwrite_flag==1，则写入
        if (not os.path.exists(json_full_path)) or (overwrite_flag == 1): 
            with open(json_full_path, mode="w", encoding='utf-8') as f:
                json.dump(sample_json_dict, f, indent=2)
            if sampleNO%1000==999:
                print('!!!'+json_full_path+' saved!!!')
        eeg = sample_time_electrode[sampleNO, :, :]
        npy_fname = sampleNO_str+'.npy'
        npy_full_path = os.path.join(divide_save_path, npy_fname)
        # 如果文件尚未被创建 或者 overwrite_flag==1，则写入
        if (not os.path.exists(npy_full_path)) or (overwrite_flag == 1): 
            np.save(npy_full_path,eeg.astype(np.float32))

            if sampleNO%1000==999:
                print('!!!'+npy_full_path+' saved!!!')















