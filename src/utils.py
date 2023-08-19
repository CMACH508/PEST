import pickle


def savepickle(data, pickle_path):
    pickle.dump(data, open(pickle_path,'wb'))


def readpickle(pickle_path):
    data = pickle.load(open(pickle_path,'rb'))
    return data
