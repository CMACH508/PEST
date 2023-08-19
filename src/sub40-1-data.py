import os
import json
from utils import readpickle, savepickle
import glob


file_list_json_all = readpickle('/cmach-data/liuyongchang/protein_sfc/contactlib/save/file_list_json_all.pkl')


folder_path = '/cmach-data/liuyongchang/protein_sfc/SCOPe/pdbstyle-2.08_40'  # 文件夹路径

# 获取文件夹中的所有文件
files = glob.glob(folder_path + '/**', recursive=True)
print(len(files))

file_list_json_sub40 = []
# 打印所有文件路径
for file in files:
    if os.path.isfile(file):  # 只打印文件，而不包括子文件夹
        # print(file)
        sid = os.path.splitext(os.path.basename(file))[0]
        json_path = '/cmach-data/liuyongchang/protein_sfc/contactlib/dataset/{}.json'.format(sid)
        if json_path in file_list_json_all:
            sequence = readpickle('/cmach-data/liuyongchang/protein_sfc/seq/dataset/{}.pkl'.format(sid))
            seq_len = len(sequence)
            if seq_len >= 20:
                file_list_json_sub40.append(json_path)

print(len(file_list_json_sub40))

savepickle(file_list_json_sub40, '/cmach-data/liuyongchang/protein_sfc/PSST/save/file_list_json_sub40.pkl')

'''
16117
15169
'''