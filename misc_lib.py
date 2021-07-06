import os
import random
import shutil
import time
from _testcapi import INT_MAX
from collections import Counter, OrderedDict, defaultdict
from time import gmtime, strftime
from typing import Iterable, TypeVar, Callable, Dict, List, Any, Tuple


A = TypeVar('A')
B = TypeVar('B')


def average(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


def tprint(*arg):
    tim_str = strftime("%H:%M:%S", gmtime())
    all_text = " ".join(str(t) for t in arg)
    print("{} : {}".format(tim_str, all_text))


class TimeEstimator:
    def __init__(self, total_repeat, name = "", sample_size = 10):
        self.time_analyzed = None
        self.time_count = 0
        self.total_repeat = total_repeat
        if sample_size == 10 and self.total_repeat > 10000:
            sample_size = 100
        self.name = name
        self.base = 3
        self.sample_size = sample_size
        self.progress_tenth = 1

    def tick(self):
        self.time_count += 1
        if not self.time_analyzed:
            if self.time_count == self.base:
                self.time_begin = time.time()

            if self.time_count == self.base + self.sample_size:
                elapsed = time.time() - self.time_begin
                expected_sec = elapsed / self.sample_size * self.total_repeat
                expected_min = int(expected_sec / 60)
                print("Expected time for {} : {} min".format(self.name, expected_min))
                self.time_analyzed = True
        if self.total_repeat * self.progress_tenth / 10 < self.time_count:
            print("{}0% completed".format(self.progress_tenth))
            self.progress_tenth += 1


def TEL(l: List):
    ticker = TimeEstimator(len(l))
    for e in l:
        yield e
        ticker.tick()


class CodeTiming:
    def __init__(self):
        self.acc = {}
        self.prev_tick = {}


    def tick_begin(self, name):
        self.prev_tick[name] = time.time()

    def tick_end(self, name):
        elp = time.time() - self.prev_tick[name]
        if name not in self.acc:
            self.acc[name] = 0

        self.acc[name] += elp


    def print(self):
        for key in self.acc:
            print(key, self.acc[key])


class SuccessCounter:
    def __init__(self):
        self.n_suc = 0
        self.n_total = 0

    def reset(self):
        self.n_suc = 0
        self.n_total = 0

    def suc(self):
        self.n_suc += 1
        self.n_total += 1

    def fail(self):
        self.n_total += 1

    def get_suc(self):
        return self.n_suc

    def get_total(self):
        return self.n_total

    def get_suc_prob(self):
        return self.n_suc / self.n_total

def exist_or_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def delete_if_exist(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def score_sort(list):
    return sorted(list, key = lambda x:-x[1])

def print_shape(name, tensor):
    print("{} shape : {}".format(name, tensor.shape))

def reverse_voca(word2idx):
    OOV = 1
    PADDING = 0
    idx2word = dict()
    idx2word[1] = "OOV"
    idx2word[0] = "PADDING"
    idx2word[3] = "LEX"
    for word, idx in word2idx.items():
        idx2word[idx] = word
    return idx2word

def slice_n_pad(seq, target_len, pad_id):
    coded_text = seq[:target_len]
    pad = (target_len - len(coded_text)) * [pad_id]
    return coded_text + pad



def get_textrizer(word2idx):
    idx2word = reverse_voca(word2idx)
    def textrize(indice):
        text = []
        PADDING = 0
        for i in range(len(indice)):
            word = idx2word[indice[i]]
            if word == PADDING:
                break
            text.append(word)
        return text
    return textrize

def get_textrizer_plain(word2idx):
    idx2word = reverse_voca(word2idx)
    def textrize(indice):
        text = []
        PADDING = 0
        for i in range(len(indice)):
            word = idx2word[indice[i]]
            if indice[i] == PADDING:
                break
            text.append(word)
        return " ".join(text)
    return textrize


def increment_circular(j, max_len):
    j += 1
    if j == max_len:
        j = 0
    return j


def pick1(l):
    return l[random.randrange(len(l))]


def pick2(l):
    return random.sample(l, 2)


def pair_shuffle(l):
    new_l = []
    for idx in range(0, len(l), 2):
        new_l.append( (l[idx], l[idx+1]) )

    random.shuffle(new_l)
    result = []
    for a,b in new_l:
        result.append(a)
        result.append(b)
    return result



class MovingWindow:
    def __init__(self, window_size):
        self.window_size = window_size
        self.history = []

    def append(self, average, n_item):
        all_span = self.history + [average] * n_item
        self.history = all_span[-self.window_size:]

    def append_list(self, value_n_item_list):
        for avg_val, n_item in value_n_item_list:
            self.append(avg_val, n_item)

    def get_average(self):
        if not self.history:
            return 0
        else:
            return average(self.history)


class Averager:
    def __init__(self):
        self.history = []

    def append(self, val):
        self.history.append(val)

    def get_average(self):
        if not self.history:
            return 0
        else:
            return average(self.history)


class NamedAverager:
    def __init__(self):
        self.avg_dict = defaultdict(Averager)

    def __getitem__(self, key):
        return self.avg_dict[key]

    def get_average_dict(self):
        d_out = {}
        for key, averager in self.avg_dict.items():
            d_out[key] = averager.get_average()
        return d_out





def get_first(x):
    return x[0]


def get_second(x):
    return x[1]


def get_third(x):
    return x[2]


class OpTime:


    def time_op(self, fn):
        begin = time.time()
        ret = fn()


# returns dictionary where key is the element in the iterable and the value is the func(key)


# returns dictionary where value is the func(value) of input dictionary


def flat_apply_stack(list_fn, list_of_list, verbose=True):
    item_loc = []

    flat_items = []
    for idx1, l in enumerate(list_of_list):
        for idx2, item in enumerate(l):
            flat_items.append(item)
            item_loc.append(idx1)

    if verbose:
        print("Total of {} items".format(len(flat_items)))
    results = list_fn(flat_items)

    assert len(results) == len(flat_items)

    stack = []
    cur_list = []
    line_no = 0
    for idx, item in enumerate(results):
        while idx >=0 and item_loc[idx] != line_no:
            assert len(cur_list) == len(list_of_list[line_no])
            stack.append(cur_list)
            line_no += 1
            cur_list = []
        cur_list.append(item)
    stack.append(cur_list)
    return stack


def get_dir_files(dir_path) -> List:
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for filename in filenames:
            path_list.append((os.path.join(dir_path, filename)))

    return path_list


def get_dir_files2(dir_path):
    r = []
    for item in os.scandir(dir_path):
        r.append(os.path.join(dir_path, item.name))

    return r


def get_dir_files_sorted_by_mtime(dir_path):
    path_list = get_dir_files(dir_path)
    path_list.sort(key=lambda x: os.path.getmtime(x))
    return path_list


def get_dir_dir(dir_path):
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for dirname in dirnames:
            path_list.append(os.path.join(dir_path, dirname))

    return path_list


def sample_prob(prob):
    v = random.random()
    for n, p in prob:
        v -= p
        if v < 0:
            return n
    return 1


def list_print(l, width):
    cnt = 0
    for item in l:
        print(item, end=" / ")
        cnt += 1
        if cnt == width:
            print()
            cnt = 0
    print()


class BinHistogram:
    def __init__(self, bin_fn):
        self.counter = Counter()
        self.bin_fn = bin_fn

    def add(self, v):
        self.counter[self.bin_fn(v)] += 1


class BinAverage:
    def __init__(self, bin_fn):
        self.list_dict = {}
        self.bin_fn = bin_fn

    def add(self, k, v):
        bin_id = self.bin_fn(k)
        if bin_id not in self.list_dict:
            self.list_dict[bin_id] = []

        self.list_dict[bin_id].append(v)

    def all_average(self):
        output = {}
        for k, v in self.list_dict.items():
            output[k] = average(v)
        return output


class DictValueAverage:
    def __init__(self):
        self.acc_dict = Counter()
        self.cnt_dict = Counter()

    def add(self, k, v):
        self.cnt_dict[k] += 1
        self.acc_dict[k] += v

    def avg(self, k):
        return self.acc_dict[k] / self.cnt_dict[k]

    def all_average(self) -> Dict:
        output = {}
        for k, v in self.cnt_dict.items():
            output[k] = self.avg(k)
        return output


class IntBinAverage(BinAverage):
    def __init__(self):
        super(IntBinAverage, self).__init__(lambda x: int(x))


def k_th_score(arr, k, reverse):
    return sorted(arr, reverse=reverse)[k]


def apply_threshold(arr, t):
    return [v  if v > t else 0 for v in arr]


def get_f1(prec, recall):
    if prec + recall != 0 :
        return (2 * prec * recall) / (prec + recall)
    else:
        return 0


def split_7_3(list_like):
    split = int(0.7 * len(list_like))

    train = list_like[:split]
    val = list_like[split:]
    return train, val


def file_iterator_interval(f, st, ed):
    for idx, line in enumerate(f):
        if idx < st:
            pass
        elif idx < ed:
            yield line
        else:
            break


def slice_iterator(itr, st, ed):
    for idx, item in enumerate(itr):
        if idx < st:
            pass
        elif idx < ed:
            yield item
        else:
            break


def ceil_divide(denom: int, nom: int) -> int:
    return int((denom + (nom-1)) / nom)


class DataIDGen:
    def __init__(self, start_num=0):
        self.cur_num = start_num

    def new_id(self):
        r = self.cur_num
        self.cur_num += 1
        return r


def two_digit_float(f):
    return "{0:.2f}".format(f)


def str_float_list(f_list: List[float]):
    return "[{}]".format(" ".join(map(two_digit_float, f_list)))


def timed_lmap(func: Callable[[A], B],
         iterable_something: List[A]) -> List[B]:
    r = []
    ticker = TimeEstimator(len(iterable_something))
    for e in iterable_something:
        r.append(func(e))
        ticker.tick()
    return r


class DataIDManager:
    def __init__(self, base=0, max_idx=-1):
        self.id_to_info = {}
        self.id_idx = base
        max_count = 10000
        if max_idx == -1:
            self.max_idx = base + max_count
        else:
            self.max_idx = max_idx

    def assign(self, info):
        idx = self.id_idx
        self.id_to_info[idx] = info
        self.id_idx += 1
        if self.id_idx == self.max_idx:
            print("WARNING id idx over maximum", self.max_idx)
        if self.id_idx >= INT_MAX:
            raise IndexError("id idx is larger than INT_MAX", self.max_idx)
        return idx


def bool_to_yn(label):
    return "Y" if label else "N"


def unique_list(l):
    return list(OrderedDict.fromkeys(l))


def get_duplicate_list(l):
    s = set()
    duplicate_indices = []
    for idx, e in enumerate(l):
        if e in s:
            duplicate_indices.append(idx)
        else:
            s.add(e)

    return duplicate_indices


def enum_passage(tokens: List[Any], window_size: int) -> Iterable[List[Any]]:
    cursor = 0
    while cursor < len(tokens):
        st = cursor
        ed = cursor + window_size
        second_tokens = tokens[st:ed]
        cursor += window_size
        yield second_tokens


def enum_passage_overlap(tokens: List[Any], window_size: int,
                         step_size: int,
                         break_when_touch_end) -> Iterable[List[Any]]:
    cursor = 0
    while cursor < len(tokens):
        st = cursor
        ed = cursor + window_size
        second_tokens = tokens[st:ed]
        cursor += step_size
        yield second_tokens
        if break_when_touch_end and ed >= len(tokens):
            break


def find_max_idx(itr: Iterable[A], key_fn: Callable[[A], Any]) -> int:
    max_idx = -1
    max_score = None
    for idx, entry in enumerate(itr):
        score = key_fn(entry)
        if max_idx < 0 or score > max_score:
            max_idx = idx
            max_score = score

    return max_idx


class NamedNumber(float):
    def __new__(self, value, name):
        return float.__new__(self, value)

    def __init__(self, value, extra):
        float.__init__(value)
        self.name = extra


# --- Dict related methods BEGIN ---

def get_dict_items(d: Dict[A, B], l: Iterable[A], ignore_not_found=False) -> List[B]:
    out_l = []
    for k in l:
        if ignore_not_found:
            if k in d:
                out_l.append(d[k])
        else:
            out_l.append(d[k])
    return out_l


def dict_to_tuple_list(d: Dict[A, B]) -> List[Tuple[A, B]]:
    out_l = []
    for k, v in d.items():
        out_l.append((k, v))

    return out_l


def dict_reverse(d):
    inv_map = {v: k for k, v in d.items()}
    return inv_map


def group_by(interable: Iterable[A], key_fn: Callable[[A], B]) -> Dict[B, List[A]]:
    grouped = {}
    for elem in interable:
        key = key_fn(elem)
        if key not in grouped:
            grouped[key] = list()

        grouped[key].append(elem)
    return grouped


def assign_list_if_not_exists(dict_like, key):
    if key not in dict_like:
        dict_like[key] = list()


def assign_default_if_not_exists(dict_like, key, default):
    if key not in dict_like:
        dict_like[key] = default()


def merge_dict_list(dict_list: List[Dict]) -> Dict:
    all_d = {}
    for d in dict_list:
        all_d.update(d)
    return all_d


def print_dict_tab(d):
    for key, value in d.items():
        print("{}\t{}".format(key, value))

# --- Dict related methods END ---


class BufferedWriter:
    def __init__(self):
        self.buffer = ""

    def print(self, *args):
        all_text = " ".join(str(t) for t in args)
        self.buffer += all_text

    def empty(self):
        self.buffer = ""

    def flush(self):
        print(self.buffer)
        self.buffer = ""


bw_obj = None


def bprint(*args):
    global bw_obj
    if bw_obj is None:
        bw_obj = BufferedWriter()
    bw_obj.print(*args)


def bempty():
    global bw_obj
    if bw_obj is None:
        bw_obj = BufferedWriter()
    bw_obj.empty()


def bflush():
    global bw_obj
    if bw_obj is None:
        bw_obj = BufferedWriter()
    bw_obj.flush()


def int_list_to_str(l):
    return " ".join(map(str, l))


def recover_int_list_str(s):
    assert type(s) == str
    return list(map(int, s.split()))


def readlines_strip(file_path):
    lines = open(file_path, "r").readlines()
    return list([l.strip() for l in lines])


class TimeProfiler:
    def __init__(self):
        self.cur_point = -1
        self.last_time = time.time()
        self.init_time = self.last_time
        self.time_acc = Counter()

    def check(self, point_idx):
        prev_point = self.cur_point

        now = time.time()
        time_elapsed = now - self.last_time
        interval_name = prev_point, point_idx
        self.cur_point = point_idx
        self.time_acc[interval_name] += time_elapsed
        self.last_time = now

    def print_time(self):
        for key, value in self.time_acc.items():
            print(key, value)

    def life_time(self):
        return time.time() - self.init_time


def select_one_pos_neg_doc(doc_itr: List[Any], get_label_fn=None) \
        -> Tuple[Any, Any]:
    pos_doc = []
    neg_doc = []
    if get_label_fn is None:
        def get_label_fn(x):
            return x.get_label()

    for doc in doc_itr:
        if get_label_fn(doc):
            pos_doc.append(doc)
        else:
            neg_doc.append(doc)

    if not pos_doc or not neg_doc:
        raise ValueError

    return pick1(pos_doc), pick1(neg_doc)


def split_window(items: List[Any], window_size):
    cursor = 0
    output = []
    while cursor < len(items):
        output.append(items[cursor:cursor+window_size])
        cursor += window_size

    return output