
import numpy as np

def label2logit(label, size):
    r = np.zeros([size,])
    r[label] = 1
    return r


def get_batches(data, batch_size):
    X, Y = data
    # data is fully numpy array here
    step_size = int((len(Y) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        x = []
        y = []
        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(Y):
                break
            x.append(np.array(X[idx]))
            y.append(Y[idx])
        if len(y) > 0:
            new_data.append((np.array(x), np.array(y)))

    return new_data

def get_batches_ex(data, batch_size, n_inputs):
    # data is fully numpy array here
    step_size = int((len(data) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        b_unit = [list() for i in range(n_inputs)]

        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(data):
                break
            for input_i in range(n_inputs):
                b_unit[input_i].append(data[idx][input_i])
        if len(b_unit[0]) > 0:
            batch = [np.array(b_unit[input_i]) for input_i in range(n_inputs)]
            new_data.append(batch)

    return new_data


def list_batch_grouping(data, batch_size):
    step_size = int((len(data) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        batch = []
        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(data):
                break
            batch.append(data[idx])
        new_data.append(batch)
    return new_data


def flatten_from_batches(data):
    data_len = None
    result = []
    for e in data:
        if data_len is None:
            data_len = len(e)
        batch_len = len(e[0])
        for i in range(batch_len):
            new_line = []
            for j in range(data_len):
                new_line.append(e[j][i])
            result.append(new_line)

    return result


def numpy_print(arr):
    return "".join(["{0:.3f} ".format(v) for v in arr])


def over_zero(np_arr):
    return np.less(0, np_arr).astype(np.float32)


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z