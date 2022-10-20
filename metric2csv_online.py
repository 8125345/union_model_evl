from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import re


pd.set_option("display.max_rows", None,
              "display.max_columns", None,
              "display.max_colwidth", 100,
              "display.width", 100)

if False:
    error_txt = []
    error_dataset_threshold = []
    base_path = Path('/Users/xyz/Desktop/音频数据/bgm_record_zhaoliang_temp/out_0819/out/analyser')
    delay_dir = glob(f'{base_path}/*')
    delay_list = [delay_.split('/')[-1] for delay_ in delay_dir]
    # print(delay_list)

    model_dir = glob(f'{base_path}/{delay_list[0]}/*')
    model_list = [model_.split('/')[-1] for model_ in model_dir]
    # print(sorted(model_list))



    data_threshold_dir = glob(f'{base_path}/{delay_list[0]}/{model_list[0]}/*')
    # print(sorted(data_threshold))
    for d_t in sorted(data_threshold_dir):

        dataset_ = d_t.split('/')[-1].split('_')[:-1]
        threshold = d_t.split('/')[-1].split('_')[-1]
        dataset = str(dataset_[0] + '_' + dataset_[1] + '_' + dataset_[2])
        # print(dataset)
        # print(threshold)
    metric = dict()


    for data_threshold in sorted(data_threshold_dir):
        # print(data_threshold)
        txt = data_threshold + '/predict_2_analyse_results.txt'
        # print(txt)
        if Path(txt).is_file():
            with open(txt, "r", encoding='utf-8') as t:
                dataline = t.read()
                # print(dataline)
                if dataline:
                    target_row = re.findall(r'Total num.*', dataline)
                    if target_row:
                        # print(target_row)
                        Precision = target_row[0].split(' ')[6]
                        Recall = target_row[0].split(' ')[8]
                        F1 = target_row[0].split(' ')[10].replace('-', '')
                        # print(target_row[0].split(' '))
                        key_data_threshold = Path(data_threshold).name
                        metric[f'{key_data_threshold}'] = [Precision, Recall, F1]
                        # print(Precision, Recall, F1)
                    else:
                        print('目标行不存在')
                        error_txt.append(txt)
                else:
                    print('txt无内容')
                    error_txt.append(txt)
        else:
            print('txt不存在')
            error_dataset_threshold.append(data_threshold)
    # print(metric)
    if error_txt:
        print("错误txt列表：", error_txt)
    if error_dataset_threshold:
        print("错误文件列表：", error_dataset_threshold)
    result = pd.DataFrame().from_dict(metric, orient='index').rename(columns={0:'Precision',1:'Recall', 2:'F1'})
    print(result)
    result.to_csv('./测试1.csv')


def get_metric_from_txt(batch_data_dir: str):
    error_txt = []
    error_dataset_threshold = []
    metric = dict()
    data_threshold_dir = glob(f'{batch_data_dir}/*')
    for data_threshold in sorted(data_threshold_dir):
        print(data_threshold)
        if Path(data_threshold).is_dir():
            txt = data_threshold + '/predict_2_analyse_results.txt'
            # print(txt)
            if Path(txt).is_file():
                with open(txt, "r", encoding='utf-8') as t:
                    dataline = t.read()
                    # print(dataline)
                    if dataline:
                        target_row = re.findall(r'Total num.*', dataline)
                        if target_row:
                            # print(target_row)
                            Precision = target_row[0].split(' ')[6]
                            Recall = target_row[0].split(' ')[8]
                            F1 = target_row[0].split(' ')[10].replace('-', '')
                            # print(target_row[0].split(' '))
                            key_data_threshold = Path(data_threshold).name
                            metric[f'{key_data_threshold}'] = [Precision, Recall, F1]
                            # print(Precision, Recall, F1)
                        else:
                            print('目标行不存在')
                            error_txt.append(txt)
                    else:
                        print('txt无内容')
                        error_txt.append(txt)
            else:
                print('txt不存在')
                error_dataset_threshold.append(data_threshold)

    # print(metric)
    if error_txt:
        print("错误txt列表：", error_txt)
    if error_dataset_threshold:
        print("错误文件列表：", error_dataset_threshold)
    result = pd.DataFrame().from_dict(metric, orient='index').rename(columns={0:'Precision',1:'Recall', 2:'F1'})
    # print(result)
    result.to_csv(batch_data_dir + '/' + 'metric_summary.csv', encoding='utf_8_sig')
    # return result


if __name__ == '__main__':
    base_path = Path.cwd()
    # base_path = Path('/Users/xyz/Desktop/音频数据/bgm_record_zhaoliang_temp/out_0819/out/analyser')
    # print(base_path)
    for model_dir in glob(f'{base_path}/*'):
        print(model_dir)
        if Path(model_dir).is_dir():
            get_metric_from_txt(model_dir)
    print("生成完成")






if False:
    excel_dir = './数据汇总测试.csv'
    # df = pd.read_csv('filename.csv', encoding='utf-8', index_col=0)
    data = pd.read_csv(excel_dir)


    header = pd.MultiIndex(levels=[['model1', 'model2'],
                                   ['precision', 'recall', 'F1']],
                           codes=[[0, 0, 0],
                                  [1, 1, 1]])
    result = pd.DataFrame(columns=header)
    # data.to_csv('./测试csv.csv', index=False)
    result.to_csv('./测试1.csv', index=False)
    print(result)


















