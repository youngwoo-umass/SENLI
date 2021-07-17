from functools import partial
import random
from misc_lib import pick1

G_del_factor = 0.7


def token_delete(binary_tag, x0, x1):
    assert len(x0) == len(x1)
    assert len(x0) == len(binary_tag)

    length = len(binary_tag)
    x0_new = []
    x1_new = []

    for i in range(length):
        if not binary_tag[i]:
            x0_new.append(x0[i])
            x1_new.append(x1[i])

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
    return x0_new, x1_new


def token_delete_with_indice(indice, x0, x1, x2):
    mask = [0] * len(x0)
    for idx in indice:
        mask[idx] = 1
    return token_delete(mask, x0, x1, x2)



def random_delete(num_del, x0, x1, x2):
    num_del = max(num_del, 1)
    length = len(x0)

    last_valid = 0
    for i in range(length):
        if x2[i] > 0 :
            last_valid = i
    num_del = min(num_del, last_valid)

    del_indice = random.sample(range(last_valid+1), num_del)
    x0_new = []
    x1_new = []
    x2_new = []
    mask = []

    for i in range(length):
        if i not in del_indice:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])
            mask.append(0)
        else:
            mask.append(1)

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    x_list = x0_new, x1_new, x2_new
    return x_list, mask


def seq_replace_inner(targ_mask, src_loc, x0, x1, x2):
    length = len(x0)
    x0_new = []
    x1_new = []
    x2_new = []

    f_first_del = True

    for i in range(length):
        if len(x0_new) >= length:
            break

        if not targ_mask[i]:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])
        else:
            if f_first_del:
                for idx in src_loc:
                    x0_new.append(x0[idx])
                    x1_new.append(x1[i])
                    x2_new.append(x2[i])
                    if len(x0_new) >= length:
                        break
                f_first_del = False

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)

    assert len(x0_new) == length
    assert len(x1_new) == length
    assert len(x2_new) == length
    return x0_new, x1_new, x2_new



def seq_add(src_indice, targ_loc, x0, x1, x2):
    length = len(x0)
    x0_new = []
    x1_new = []
    x2_new = []

    for i in range(length):
        if len(x0_new) >= length:
            break

        if i == targ_loc:
            for idx in src_indice:
                x0_new.append(x0[idx])
                x1_new.append(x1[i])
                x2_new.append(x2[i])
                if len(x0_new) >= length:
                    break

        x0_new.append(x0[i])
        x1_new.append(x1[i])
        x2_new.append(x2[i])


    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)

    assert len(x0_new) == length
    assert len(x1_new) == length
    assert len(x2_new) == length
    return x0_new, x1_new, x2_new


def seq_delete(num_del, info, idx_trans_fn, x0, x1, x2):
    return seq_delete_inner(num_del, x0, x1, x2)




def get_seq_deleter(g_val):
    def innner_deleter(g, num_del, x0, x1):
        length = len(x0)
        last_valid = 0
        for i in range(length):
            if x0[i] > 0 :
                last_valid = i
        num_del = min(num_del, last_valid)

        def sample_len():
            l = 1
            v = random.random()
            while v < g and l < length:
                l = l * 2
                v = random.random()
            return min(l, length)

        indice = []
        for i in range(num_del):
            del_len = sample_len()
            start_idx = pick1(range(last_valid+1))
            end_idx = min(start_idx+del_len, last_valid+1)
            for idx in range(start_idx, end_idx):
                indice.append(idx)

        mask = [0] * len(x0)
        for idx in indice:
            mask[idx] = 1

        return token_delete(mask, x0, x1), mask
    return partial(innner_deleter, g_val)


def seq_delete_inner(num_del, x0, x1, x2):
    length = len(x0)
    last_valid = 0
    for i in range(length):
        if x2[i] > 0 :
            last_valid = i
    num_del = min(num_del, last_valid)

    def sample_len():
        l = 1
        v = random.random()
        while v < G_del_factor and l < length:
            l = l * 2
            v = random.random()
        return min(l, length)

    indice = []
    for i in range(num_del):
        del_len = sample_len()
        start_idx = pick1(range(last_valid+1))
        end_idx = min(start_idx+del_len, last_valid+1)
        for idx in range(start_idx, end_idx):
            indice.append(idx)

    mask = [0] * len(x0)
    for idx in indice:
        mask[idx] = 1

    return token_delete(mask, x0, x1, x2), mask


def random_delete_with_mask(num_del, x0, x1, x2, q_mask):
    num_del = max(num_del, 1)
    length = len(x0)

    sample_space = []
    for i in range(length):
        if q_mask[i] > 0 :
            sample_space.append(i)
    num_del = min(num_del, len(sample_space))

    del_indice = random.sample(sample_space, num_del)
    x0_new = []
    x1_new = []
    x2_new = []
    mask = []

    for i in range(length):
        if i not in del_indice:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])
            mask.append(0)
        else:
            mask.append(1)

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    x_list = x0_new, x1_new, x2_new
    return x_list, mask


def random_replace_with_mask(num_del, x0, x1, x2, q_mask, random_token):
    num_del = max(num_del, 1)
    length = len(x0)

    sample_space = []
    for i in range(length):
        if q_mask[i] > 0 :
            sample_space.append(i)
    num_del = min(num_del, len(sample_space))

    del_indice = random.sample(sample_space, num_del)
    x0_new = []
    x1_new = []
    x2_new = []
    mask = []

    for i in range(length):
        if i not in del_indice:
            x0_new.append(x0[i])
            x1_new.append(x1[i])
            x2_new.append(x2[i])
            mask.append(0)
        else:
            x0_new.append(random_token())
            x1_new.append(x1[i])
            x2_new.append(x2[i])
            mask.append(1)

    while len(x0_new) < len(x0):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    x_list = x0_new, x1_new, x2_new
    return x_list, mask