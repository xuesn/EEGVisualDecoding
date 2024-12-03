
import os
import mne  
import math
import json
import numpy as np
import pandas as pd
from colorama import init, Fore,  Back,  Style
init(autoreset = True)

overwrite_flag = 0
# 预处理参数
ori_data_amplify_rate= 1
amplify_rate=1_000_000
# 频域

# 62.5Hz
downsample_freq = 62.5
freq_low = None
freq_high = None
freq_notch = None



epoch_st =  0.0# 以s为单位
epoch_end = 0.5 # 以s为单位
# 基线校正
baseline_st = 0
baseline_end = 0

# 单样本  KB
# ----------------------------------------------------------------------------------------------------
#一些记录信息
dataset_str='18_Austria'
data_type=''
paradigm_type='RSVP'
# 切分后存储位置

# 62.5Hz
save_dir=''
save_dir=save_dir.replace('.','p')
save_dir=save_dir.replace(' ','_')


# load
author_data_dir=''
file_list=os.listdir(author_data_dir)
sub_list = [fn for fn in file_list if fn.startswith('sub-')]
sub_list.sort()
sub_st = 1
sub_end = 16
for _,sub_dir in enumerate(sub_list[sub_st-1:sub_end]):
    sub_data_path = os.path.join(author_data_dir,sub_dir,'eeg')
    file_list=os.listdir(sub_data_path)
    vhdr_list = [fn for fn in file_list if fn.endswith('.vhdr')]
    vhdr_list.sort() 
    #break

    for data_fname in vhdr_list:       


        print('开始预处理',sub_dir,data_fname)


        label_fname = data_fname.replace('eeg.vhdr','events.tsv')
        label_path = os.path.join(sub_data_path, label_fname)
        sub_rsvp_events = pd.read_csv(label_path, sep='\t')
       

        sub_str = data_fname[:6]
        if 'run-01' in data_fname:
            data_type = '5Hz'
        elif 'run-02' in data_fname:
            data_type = '20Hz'
        save_path = os.path.join(save_dir,data_type,sub_str)



        sample_num =   len(sub_rsvp_events)
        sample_num_after_delete =  8000 #删去了boats和star

        # 标签等信息
        stimulusname=sub_rsvp_events.stimulusname
        stimulusnumber=sub_rsvp_events.stimulusnumber
        istarget=sub_rsvp_events.istarget
        trialnumber=sub_rsvp_events.trialnumber
        condition=sub_rsvp_events.condition
        levelA=sub_rsvp_events.levelA
        levelB=sub_rsvp_events.levelB
        levelC=sub_rsvp_events.levelC
        withinexemplarnumber=sub_rsvp_events.withinexemplarnumber
        response=sub_rsvp_events.response
        correct=sub_rsvp_events.correct
        rt=sub_rsvp_events.rt
        {
            "stimulusname":"path to the stimulus",
            "stimulusnumber":"stimulus number (1:200 are stimuli, 201:216 are targets)",
            "istarget":"presented stimulus was a target",
            "trialnumber":"sequence number",
            "condition":"whether participant was looking for boats or stars",
            "levelA":"category at highest level (animate/inanimate)",
            "levelB":"category at second level (furniture, fruit, mammal)",
            "levelC":"category at third level (table, cow, apple)",
            "withinexemplarnumber":"image number within level C (4 images per levelC category",
            "response":"number of targets the participant reported seeing in the stream",
            "correct":"whether the response matches the number of targets",
            "rt":"reaction time"
        }
        #我这里没按字母顺序，而是按类别排的
        levelA_list = ['animate', 'inanimate']
        levelB_list = ['aquatic', 'bird', 'human', 'insect', 'mammal', 'clothing', 'fruits', 'furniture', 'plants', 'tools']
        levelC_list = [
            'crab', 'fish', 'seahorse', 'shark', 'shrimp', 
            'chicken', 'duck', 'parrot', 'peacock', 'penguin', 
            'baby', 'child', 'clown', 'faces', 'sports', 
            'ant', 'bug', 'butterfly', 'dragonfly', 'grasshopper', 
            'cow', 'giraffe', 'kangaroo', 'monkey', 'rhino', 
            'gloves', 'hat', 'jeans', 'shirt', 'socks', 
            'apple', 'banana', 'grape', 'pineapple', 'strawberry', 
            'bed', 'chair', 'closet', 'sofa', 'table', 
            'bush', 'cactus', 'sunflower', 'tree', 'tulip', 
            'broom', 'hammer', 'pan', 'scissors', 'wrench']
        target_list = ['boats',  'star']    #作者的失误吧，boat加了复数
        class_list = levelC_list+target_list
        sample_label_0_x=np.array([class_list.index(i) for i in levelC])
        sample_img_str=np.array(stimulusname)
        sample_info_list=[]
        for i in range(sample_num):
            info='stimulusname-'+str(stimulusname[i]) +\
                '__stimulusnumber-'+str(stimulusnumber[i]) +\
                '__istarget-'+str(istarget[i]) +\
                '__trialnumber-'+str(trialnumber[i]) +\
                '__condition-'+str(condition[i]) +\
                '__levelA-'+str(levelA[i]) +\
                '__levelB-'+str(levelB[i]) +\
                '__levelC-'+str(levelC[i]) +\
                '__withinexemplarnumber-'+str(withinexemplarnumber[i]) +\
                '__response-'+str(response[i]) +\
                '__correct-'+str(correct[i]) +\
                '__reacttime-'+str(rt[i])
            sample_info_list.append(info)
        sample_info=np.array(sample_info_list)
        # - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ 
        #读数据和预处理
        data_path = os.path.join(sub_data_path, data_fname)
        raw_data = mne.io.read_raw_brainvision(data_path)
        # break   
        raw_data.ch_names #len=63
        # 63-channel electro encephalography, 

        raw_data.load_data()
        # 重设采样率 
        raw_data.resample(sfreq=downsample_freq) 

        # - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ 
        # onset，图片展示时刻
        events, event_id = mne.events_from_annotations(raw_data) 
        event_value_list = events[:, 2].tolist()

        event_id_epoch = [10001]

        # 获取label
        target_event = event_id_epoch
        target_event_value_list=[i for i in event_value_list if i in target_event]

        # 有的10001比样本数多1个，根据观察第一个10001时间间隔很长，应该代表开始，故丢弃
        events_list=events.tolist()
        target_events_list=[[j,k,i] for (j,k,i) in events_list if i in target_event]
        #替换events
        events = np.array(target_events_list)
        print(Fore.RED+str(events[:5,]))
        if (events[1,0] - events[0,0]) > 0.2*downsample_freq + 10:  #若大于210ms 则认为是错打的，删去
            events = events[1:,]

        electrode_num = len(raw_data.ch_names)
        # - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ 
        # 切分
        epochs = mne.Epochs(raw_data, events, event_id_epoch, tmin=epoch_st, tmax=epoch_end,
                            baseline=(baseline_st, baseline_end), picks=None, preload=False,
                            reject=None, flat=None, proj=True, decim=1,
                            reject_tmin=None, reject_tmax=None, detrend=None,
                            on_missing='raise', reject_by_annotation=True, metadata=None,
                            event_repeated='error', verbose=None)
        # 转np数组
        sample_time_electrode = epochs.get_data().transpose(0, 2, 1)
        # mne.Epochs会多切1个值，因此去除 (据检查应该是后面多一个，也差不了多少)
        sample_time_electrode = sample_time_electrode[:, :-1, :]
        # 若cpu内存不足，就可以del
        # del epochs
        print('sample_time_electrode', 'mean:', np.mean(sample_time_electrode))
        print('sample_time_electrode', 'max:', np.max(sample_time_electrode))
        print('sample_time_electrode', 'min:', np.min(sample_time_electrode))
        print('sample_time_electrode', 'std:', np.std(sample_time_electrode))
        print('sample_time_electrode-shape:', sample_time_electrode.shape)
    # ----------------------------------------------------------------------------------------------------
        #去除boats stars等不参与分类的样本
        delete_sample_index = sample_label_0_x>49       #第51、52类
        
        sample_label_0_x=sample_label_0_x[~delete_sample_index]
        sample_img_str=sample_img_str[~delete_sample_index]
        sample_info=sample_info[~delete_sample_index]
        sample_time_electrode=sample_time_electrode[~delete_sample_index,:,:]        
        
        '''
        # aaa=raw_data.load_data()
        bbb=raw_data.get_data()
        np.max(bbb)
        np.min(bbb)
        np.mean(bbb)
        import matplotlib.pyplot as plt
        plt.plot(bbb[40,50000:50000+300000:100])
        '''

    # ----------------------------------------------------------------------------------------------------
        # 保存
        sample_info_rear = data_fname + ' ' +\
            str(downsample_freq)+'Hz' + ' ' +\
            'bp' + str(freq_low)+'_' + str(freq_high)+'Hz' + ' ' +\
            'notch' + str(freq_notch)+'Hz' + ' ' +\
            str(int(epoch_st*1000))+'_' + str(int(epoch_end*1000))+'ms' + ' ' +\
            'bc' + str(int(baseline_st*1000))+'_' + str(int(baseline_end*1000))+'ms' + ' ' +\
            'ori_data_amplify_rate' + str(ori_data_amplify_rate)+' ' +\
            'amplify_rate' + str(amplify_rate)+' ' +\
            '.json'
        # 逐样本保存
        sample_num = sample_time_electrode.shape[0]
        if sample_time_electrode.shape[0] != len(sample_label_0_x):
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
            label = sample_label_0_x[sampleNO]
            img_str=sample_img_str[sampleNO]
            info = sample_info[sampleNO]
            sample_json_dict = {
                "label": int(label),
                "image": img_str,
                "info": info,
            }
            #存标签等信息
            json_fname = sampleNO_str+'.json'
            json_full_path = os.path.join(divide_save_path, json_fname)
            # 如果文件尚未被创建 或者 overwrite_flag==1，则写入
            overwrite_flag = 0
            if (not os.path.exists(json_full_path)) or (overwrite_flag == 1): 
                with open(json_full_path, "w") as f:
                    json.dump(sample_json_dict, f, indent=2)
                if sampleNO%1000==999:
                    print('!!!'+json_full_path+' saved!!!')
            eeg = sample_time_electrode[sampleNO, :, :]*amplify_rate/ori_data_amplify_rate #乘上amplify_rate，避免精度损失
            npy_fname = sampleNO_str+'.npy'
            npy_full_path = os.path.join(divide_save_path, npy_fname)
            # 如果文件尚未被创建 或者 overwrite_flag==1，则写入
            if (not os.path.exists(npy_full_path)) or (overwrite_flag == 1): 
                #存为float32，应该能节省一半的空间
                np.save(npy_full_path,eeg.astype(np.float32))
                if sampleNO%1000==999:
                    print('!!!'+npy_full_path+' saved!!!')



