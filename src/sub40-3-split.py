import random
from utils import readpickle, savepickle
from collections import Counter


file_list = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/file_list_json_sub40.pkl')
sid2label = readpickle('/cmach-data/liuyongchang/protein_sfc/contactlib/save/sid2label.pkl')
sid_list_207 = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/sid_list_207.pkl')

train_file = []
val_file = []
test_file = []

record = {}
for file in file_list:
    sid = file.split('/')[-1][:7]
    if sid not in sid_list_207:
        test_file.append(file)
        continue
    superfamily = sid2label[sid]['superfamily']
    if superfamily not in record:
        record[superfamily] = [file]
    else:
        record[superfamily].append(file)

n_superfamily = len(list(record.keys()))
print(n_superfamily)


for superfamily, file_list in record.items():
    random.shuffle(file_list)
    train_file.append(file_list[0])
    l = len(file_list) - 1
    val_l = int(l * 0.2)
    train_l = l - val_l
    for i in range(1, val_l+1):
        val_file.append(file_list[i])
    for i in range(val_l+1, l+1):
        train_file.append(file_list[i])


print(len(train_file))
print(len(val_file))
print(len(test_file))

dataset_split = {'train_file': train_file, 'val_file':val_file, 'test_file': test_file}
savepickle(dataset_split, '/cmach-data/liuyongchang/protein_sfc/PSST/save/dataset_split_sub40.pkl')


'''
1961
11605
1873
1691
'''