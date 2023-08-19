import random
from utils import readpickle, savepickle
from collections import Counter
import matplotlib.pyplot as plt


file_list_json = readpickle('/cmach-data/liuyongchang/protein_sfc/PSST/save/file_list_json_sub40.pkl')
sid2label = readpickle('/cmach-data/liuyongchang/protein_sfc/contactlib/save/sid2label.pkl')

class_list = []
fold_list = []
superfamily_list = []

for file in file_list_json:
    sid = file.split('/')[-1][:7]
    pclass = sid2label[sid]['class']
    fold = sid2label[sid]['fold']
    superfamily = sid2label[sid]['superfamily']
    class_list.append(pclass)
    fold_list.append(fold)
    superfamily_list.append(superfamily)


counter = Counter(superfamily_list)
count_list = []
superfamily_1s_list = []
superfamily_count = {}
for element, count in counter.items():
    count_list.append(count)
    superfamily_count[element] = count
    if count == 1:
        superfamily_1s_list.append(element)



savepickle(superfamily_count, '/cmach-data/liuyongchang/protein_sfc/PSST/save/superfamily_count_sub40.pkl')

# print('number of superfamily: {}'.format(len(count_list)))

counter2 = Counter(count_list)
x = []
y = []
for element, count in counter2.items():
    x.append(element)
    y.append(count)
    # print(element, count)


sorted_xy = sorted(zip(x, y))

sorted_x, sorted_y = zip(*sorted_xy)

# for i in range(len(x)):
#     print(sorted_x[i], sorted_y[i])


# print("len(superfamily_1s_list):", len(superfamily_1s_list))


print('class')
class_list = list(set(class_list))
print(len(class_list))
class_list.sort()
# print(superfamily_list)

class2idx = {}
for i in range(len(class_list)):
    pclass = class_list[i]
    class2idx[pclass] = i

savepickle(class2idx, '/cmach-data/liuyongchang/protein_sfc/PSST/save/class2idx_sub40.pkl')


print('fold')
fold_list = list(set(fold_list))
print(len(fold_list))
fold_list.sort()
# print(superfamily_list)

fold2idx = {}
for i in range(len(fold_list)):
    fold = fold_list[i]
    fold2idx[fold] = i

savepickle(fold2idx, '/cmach-data/liuyongchang/protein_sfc/PSST/save/fold2idx_sub40.pkl')


print('superfamily')
superfamily_list = list(set(superfamily_list))
print(len(superfamily_list))
superfamily_list.sort()
# print(superfamily_list)

lab2idx = {}
for i in range(len(superfamily_list)):
    lab = superfamily_list[i]
    lab2idx[lab] = i

savepickle(lab2idx, '/cmach-data/liuyongchang/protein_sfc/PSST/save/lab2idx_sub40.pkl')


'''
class
7
fold
1257
superfamily
2065
'''