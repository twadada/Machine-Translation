import numpy as np


def Generate_bacth_idx(dataset, batch_size):
    #dataset: sorted by src length
    batch_idx_list = []
    idx= [np.asarray(np.argwhere(np.array(dataset.lengths[0]) == i)).reshape(-1) for i in
                   set(dataset.lengths[0])]
    for i in range(len(idx)):
        idx_shuffled = np.random.permutation(idx[i]) #shuffle batch idx within idx grouped by length
        idx_i_len = len(idx_shuffled) #size of groped idx by sentence length
        batch_len = (idx_i_len-1) // batch_size # N of splits (subtract 1, as 4//2 = 2 but element_size_ary should be [2]  )
        if(batch_len == 0): #if idx_i_len < batch_size
            array_split = [np.array(idx_shuffled)] #add one minibatch (size < batch_size)
        else:
            element_size_ary = [batch_size] * batch_len
            array_split = np.array_split(idx_shuffled, np.cumsum(element_size_ary))

        batch_idx_list.extend(array_split)

    return batch_idx_list

def Sort_batch_by_srclen(dataset):

        idx = np.argsort(dataset.lengths[0])
        for i in range(2):
            dataset.lines_id[i] = [dataset.lines_id[i][j] for j in idx]
            dataset.lengths[i] = dataset.lengths[i][idx]
        return dataset
        #return idx_grouped, idx

def Shuffle_train_data(dataset):
        data_size = int(len(dataset.lines_id[0]))
        # shuffle training data
        sffindx = np.random.permutation(data_size)
        for i in range(2):
            dataset.lines[i] = [dataset.lines[i][j] for j in sffindx]
            dataset.lines_id[i] = [dataset.lines_id[i][j] for j in sffindx]
            dataset.lengths[i] = dataset.lengths[i][sffindx]

        return dataset
