import copy
import csv

from cache import *
from data_generator.NLI.enlidef import *
from data_generator.NLI.nli_info import corpus_dir
from data_generator.data_parser.esnli import load_split
from data_generator.subword_translate import normalize_pt
from data_generator.text_encoder import SubwordTextEncoder, CLS_ID, SEP_ID
from data_generator.tokenizer_b import FullTokenizerWarpper, _truncate_seq_pair
from data_generator.tokenizer_wo_tf import FullTokenizer
from evaluation import *
from path_manager import bert_voca_path


def get_modified_data_loader2(hp):
    tokenizer = FullTokenizer(bert_voca_path)
    data_loader = DataLoader(hp.seq_max)
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    data_loader.CLS_ID = CLS_ID
    data_loader.SEP_ID = SEP_ID
    return data_loader


def get_modified_data_loader(tokenizer, max_sequence):
    data_loader = DataLoader(max_sequence)
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    data_loader.CLS_ID = CLS_ID
    data_loader.SEP_ID = SEP_ID
    return data_loader


class DataLoader:
    def __init__(self, max_sequence, load_both_dev=False):
        self.train_data = None
        self.dev_data = None
        self.test_data = None

        self.train_file = os.path.join(corpus_dir, "train.tsv")
        self.dev_file = os.path.join(corpus_dir, "dev_matched.tsv")
        self.dev_file2 = os.path.join(corpus_dir, "dev_mismatched.tsv")
        self.max_seq = max_sequence
        self.load_both_dev = load_both_dev
        self.CLS_ID = CLS_ID
        self.SEP_ID = SEP_ID
        voca_path = bert_voca_path
        assert os.path.exists(voca_path)
        self.name = "nli"
        self.lower_case = True
        self.sep_char = "#"
        self.encoder = FullTokenizerWarpper(voca_path)

        self.dev_explain_0 = None
        self.dev_explain_1 = None

    def get_train_data(self):
        if self.train_data is None:
            self.train_data = load_cache("nli_train_cache")

        if self.train_data is None:
            data = list(self.example_generator(self.train_file))
            self.train_data = data
        save_to_pickle(self.train_data, "nli_train_cache")
        return self.train_data

    def get_dev_data(self):
        if self.dev_data is None:
            self.dev_data = load_cache("nli_dev_cache")

        if self.dev_data is None:
            self.dev_data = list(self.example_generator(self.dev_file))
            if self.load_both_dev:
                self.dev_data += list(self.example_generator(self.dev_file2))
        save_to_pickle(self.dev_data, "nli_dev_cache")
        return self.dev_data

    def get_train_infos(self):
        infos = list(self.info_generator(self.train_file))
        return infos

    def get_dev_infos(self):
        return list(self.info_generator(self.dev_file))

    def get_dev_explain(self, target):
        if target == 'conflict':
            return self.get_dev_explain_0()
        elif target == 'match' or target == 'mismatch':
            return self.get_dev_explain_1(target)
        else:
            assert False

    def get_dev_explain_0(self):
        if self.dev_explain_0 is None:
            explain_data = load_mnli_explain_0()

            def entry2inst(raw_entry):
                entry = self.encode(raw_entry['p'], raw_entry['h'])
                return entry["input_ids"], entry["input_mask"], entry["segment_ids"]

            encoded_data = list([entry2inst(entry) for entry in explain_data])
            self.dev_explain_0 = encoded_data, explain_data

        return self.dev_explain_0

    def get_dev_explain_1(self, tag):
        if self.dev_explain_1 is None:
            explain_data = list(load_nli_explain_1(tag))

            def entry2inst(raw_entry):
                entry = self.encode(raw_entry[0], raw_entry[1])
                return entry["input_ids"], entry["input_mask"], entry["segment_ids"]

            encoded_data = list([entry2inst(entry) for entry in explain_data])
            self.dev_explain_1 = encoded_data, explain_data
        return self.dev_explain_1

    def get_test_data(self, data_id):
        if data_id.startswith("test_"):
            encoded_data, plain_data = self.load_plain_text(data_id)
        else:
            if data_id == 'conflict':
                data = self.get_dev_explain_0()
                encoded_data, explain_data = data
                plain_data = list([(entry['p'], entry['h']) for entry in explain_data])
            elif data_id == 'match':
                data = self.get_dev_explain('match')
                encoded_data, explain_data = data
                plain_data = list([(entry[0], entry[1]) for entry in explain_data])

        return encoded_data, plain_data

    def load_plain_text(self, data_id):
        data = load_plain_text(data_id + ".csv")

        def entry2inst(raw_entry):
            entry = self.encode(raw_entry[0], raw_entry[1])
            return entry["input_ids"], entry["input_mask"], entry["segment_ids"]

        encoded_data = list([entry2inst(entry) for entry in data])
        return encoded_data, data

    # enc_explain_dev = list[ input_ids, input_mask, segment_ids]
    # explain_dev = list[prem, hypo, p_indice, h_indice]
    def match_explain_info(self, enc_explain_dev, explain_dev):
        info_list = self.get_dev_infos()

        def equal(e1, e2):
            return e1.strip().lower() == e2.strip().lower()

        def similar(e1, e2):
            def feature(s):
                return set(s.lower().strip().split())

            n_common = len(feature(e1).intersection(feature(e2)))
            return n_common / len(feature(e1))

        result = []
        for enc_entry, plain_entry in list(zip(enc_explain_dev, explain_dev)):
            prem, hypo, _, _ = plain_entry

            matching = None
            for info_entry in info_list:
                s1, s2, bp1, bp2 = info_entry
                if equal(s1, prem) and equal(s2, hypo):
                    matching = info_entry
                    break


            if matching is None:
                entries = []
                for info_entry in info_list:
                    s1, s2, bp1, bp2 = info_entry

                    score1 = similar(s1, prem)
                    score2 = similar(s2, hypo)
                    if score1 > 0.5 or score2 > 0.5:
                        entries.append((info_entry, score1*score2))

                entries.sort(key=lambda x:x[1], reverse=True)
                matching = entries[0][0]


            assert matching is not None
            result.append((enc_entry, matching))
        return result


    def convert_index_out(self, raw_sentence, subtoken_ids, target_idx):
        if self.lower_case:
            raw_sentence = raw_sentence.lower()
        tokens = raw_sentence.split()
        subword_tokens = self.encoder.decode_list(subtoken_ids)
        # print("subword_tokens", subword_tokens)
        # print("target subword", subword_tokens[target_idx])
        if subword_tokens[target_idx].replace("_", "").replace(" ", "") == "":
            target_idx = target_idx - 1
            # print("Replace target_idx to previous", subword_tokens[target_idx])
        prev_text = "".join(subword_tokens[:target_idx])
        text_idx = 0
        # print("prev text", prev_text)
        # now we want to find a token from raw_sentence which appear after prev_text equivalent

        def update_text_idx(target_char, text_idx):
            while prev_text[text_idx] in [self.sep_char, " "]:
                text_idx += 1
            if target_char == prev_text[text_idx]:
                text_idx += 1
            return text_idx

        try:
            for t_idx, token in enumerate(tokens):
                for c in token:
                    # Here, previous char should equal prev_text[text_idx]
                    text_idx = update_text_idx(c, text_idx)
                    # Here, c should equal prev_text[text_idx-1]
                    assert c == prev_text[text_idx-1]

        except IndexError:
            #print("target_token", tokens[t_idx])
            #print("t_idx", t_idx)
            return t_idx
        raise Exception

    def convert_indice_in(self, tokens, input_x, indice, seg_idx):
        sub_tokens = self.split_p_h(input_x[0], input_x)
        subword_tokens = self.encoder.decode_list(sub_tokens[seg_idx])
        start_idx = [1, 1 + len(sub_tokens[0]) + 1][seg_idx]
        in_segment_indice = translate_index(tokens, subword_tokens, indice)
        return list([start_idx + idx for idx in in_segment_indice])


    def test(self):
        sent = "Nonautomated First-Class and Standard-A mailers cannot ask for their mail to be processed by hand because it costs the postal service more."
        subtoken_ids = self.encoder.encode(sent)
        print(self.encoder.decode_list(subtoken_ids))

    def class_labels(self):
        return ["entailment", "neutral", "contradiction",]

    def get_genres(self, filename):
        genres = set()
        for idx, line in enumerate(open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Works for both splits even though dev has some extra human labels.
            genres.add(split_line[3])
        return genres

    def get_train_genres(self):
        return self.get_genres(self.train_file)

    def example_generator_w_genre(self, filename, target_genre):
        label_list = self.class_labels()
        for idx, line in enumerate(open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Works for both splits even though dev has some extra human labels.
            s1, s2 = split_line[8:10]
            genre = split_line[3]
            if genre == target_genre:
                l = label_list.index(split_line[-1])
                entry = self.encode(s1, s2)
                yield entry["input_ids"], entry["input_mask"], entry["segment_ids"], l

    def get_raw_example(self, filename, target_genre ):
        label_list = self.class_labels()
        for idx, line in enumerate(open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Works for both splits even though dev has some extra human labels.
            s1, s2 = split_line[8:10]
            genre = split_line[3]
            if genre == target_genre:
                l = label_list.index(split_line[-1])
                yield s1, s2, l

    def example_generator(self, filename):
        label_list = self.class_labels()
        for idx, line in enumerate(open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Works for both splits even though dev has some extra human labels.
            s1, s2 = split_line[8:10]
            l = label_list.index(split_line[-1])
            entry = self.encode(s1, s2)

            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"], l

    def info_generator(self, filename):
        label_list = self.class_labels()
        for idx, line in enumerate(open(filename, "rb")):
            if idx == 0: continue  # skip header
            line = line.strip().decode("utf-8")
            split_line = line.split("\t")
            # Works for both splits even though dev has some extra human labels.
            s1, s2 = split_line[8:10]
            bp1, bp2 = split_line[4:6] # bp : Binary Parse
            l = label_list.index(split_line[-1])
            yield s1, s2, bp1, bp2

    # split the np_arr, which is an attribution scores
    def split_p_h(self, np_arr, input_x):
        input_ids, _, seg_idx = input_x
        return self.split_p_h_with_input_ids(np_arr, input_ids)

    def split_p_h_with_input_ids(self, np_arr, input_ids):

        for i in range(len(input_ids)):
            if input_ids[i] == self.SEP_ID:
                idx_sep1 = i
                break

        p = np_arr[1:idx_sep1]
        for i in range(idx_sep1 + 1, len(input_ids)):
            if input_ids[i] == self.SEP_ID:
                idx_sep2 = i
        h = np_arr[idx_sep1 + 1:idx_sep2]
        return p, h

    def encode(self, text_a, text_b):
        tokens_a = self.encoder.encode(text_a)
        tokens_b = self.encoder.encode(text_b)

        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq - 3)

        tokens = []
        segment_ids = []
        tokens.append(self.CLS_ID)
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(self.SEP_ID)
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append(self.SEP_ID)
            segment_ids.append(1)

        input_ids = tokens

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq
        assert len(input_mask) == self.max_seq
        assert len(segment_ids) == self.max_seq

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids
        }


class SNLIDataLoader(DataLoader):
    def __init__(self, max_sequence, vocab_filename):
        super(SNLIDataLoader, self).__init__(max_sequence, vocab_filename)
        self.name = "snli"

    def get_train_data(self):
        if self.train_data is None:
            self.train_data = list(self.example_generator("train"))
        return self.train_data

    def get_dev_data(self):
        if self.dev_data is None:
            self.dev_data = list(self.example_generator("dev"))
        return self.dev_data

    def example_generator(self, split):
        label_list = self.class_labels()
        r = load_split(split)
        for e in r:
            l = label_list.index(e['gold_label'])
            entry = self.encode(e['Sentence1'], e['Sentence2'])

            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"], l

    def load_plain_text(self, name):
        data = load_split(name)

        def get_plain(entry):
            return entry['Sentence1'], entry['Sentence2']

        def entry2inst(raw_entry):
            entry = self.encode(raw_entry['Sentence1'], raw_entry['Sentence2'])
            return entry["input_ids"], entry["input_mask"], entry["segment_ids"]

        encoded_data = list([entry2inst(entry) for entry in data])
        plain_entry = list([get_plain(e) for e in data])
        return encoded_data, plain_entry


# Output : Find indice of subword_tokens that covers indice of parse_tokens
# Lowercase must be processed
# indice is for parse_tokens
def translate_index(parse_tokens, subword_tokens, indice):
    print_buf = ""
    result = []
    def dbgprint(s):
        nonlocal print_buf
        print_buf += s + "\n"
        
    try:
        sep_char = "#"

        def normalize_pt_str(s):
            if "&" in s and "&amp;" not in s and "& amp" not in s:
                sw_str = "".join(subword_tokens)
                if "&amp;" in sw_str:
                    s = s.replace("&", "&amp;")
            return s

        def normalize_sw(text):
            s = text.replace("”", "\"").replace("“", "\"").replace("…", "...").replace("«", "\"")
            if s == "#":
                s = "shop;"
            return s


        parse_tokens = list([normalize_pt(t.lower()) for t in parse_tokens])
        subword_tokens = list([normalize_sw(t) for t in subword_tokens])
        dbgprint("----")
        dbgprint("parse_tokens : " + " ".join(parse_tokens))
        dbgprint("subword_tokens : " + " ".join(subword_tokens))
        for target_index in indice:
            if target_index > len(subword_tokens):
                break
            pt_begin = normalize_pt_str("".join(parse_tokens[:target_index]))

            pt_idx = 0
            dbgprint("Target_index {}".format(target_index))
            dbgprint("prev_text : " + pt_begin)

            # find the index in subword_tokens that covers

            # Step 1 : Find begin of parse_tokens[target_idx]
            swt_idx = 0
            sw_idx = 0
            while pt_idx < len(pt_begin):
                token = subword_tokens[swt_idx]
                if sw_idx == 0:
                    dbgprint(token)

                c = token[sw_idx]
                if c in [sep_char, " "]:
                    sw_idx += 1
                elif c == pt_begin[pt_idx]:
                    sw_idx += 1
                    pt_idx += 1
                    assert c == pt_begin[pt_idx - 1]
                else:
                    raise Exception("Non matching 1 : {} not equal {}".format(c, pt_begin[pt_idx]) )
                    assert False

                if sw_idx == len(token): # Next token
                    swt_idx += 1
                    sw_idx = 0

            while subword_tokens[swt_idx][sw_idx] in [sep_char, " "]:
                sw_idx += 1

            dbgprint("")
            if subword_tokens[swt_idx] == "[UNK]":
                continue
            assert pt_idx == len(pt_begin)
            if not parse_tokens[target_index][0] == subword_tokens[swt_idx][sw_idx]:
                print("swt_idx = {} sw_idx= {}".format(swt_idx, sw_idx))
                raise Exception("Non matching 2 : {} not equal {}".format(parse_tokens[target_index][0], subword_tokens[swt_idx][sw_idx]) )
            # Step 2 : Add till parse_tokens[target_idx] ends
            pt_end = normalize_pt_str("".join(parse_tokens[:target_index + 1]))
            dbgprint("pt_end : " + pt_end)
            while pt_idx < len(pt_end):
                token = subword_tokens[swt_idx]
                if len(result) == 0 or result[-1] != swt_idx:
                    dbgprint("Append {} ({})".format(swt_idx, token))
                    result.append(swt_idx)

                c = token[sw_idx]
                if c in [sep_char, " "]:
                    sw_idx += 1
                elif c == pt_end[pt_idx]:
                    sw_idx += 1
                    pt_idx += 1
                    assert c == pt_end[pt_idx - 1]
                else:
                    raise Exception("Non matching 3 : {} not equal {}".format(c, pt_begin[pt_idx]))
                    assert False

                if sw_idx == len(token): # Next token
                    swt_idx += 1
                    sw_idx = 0
    except Exception as e:
        print(e)
        print(print_buf)
    return result


def load_nli_explain():
    path = os.path.join(corpus_dir, "conflict.csv")
    f = open(path, "r")
    reader = csv.reader(f, delimiter=',')

    for idx, row in enumerate(reader):
        if idx ==0 : continue
        premise = row[0]
        hypothesis= row[1]
        tokens_premise = row[2].split()
        tokens_hypothesis= row[3].split()

        for t in tokens_hypothesis:
            if t.lower() not in hypothesis.lower():
                raise Exception(t)
        for t in tokens_premise:
            if t.lower() not in premise.lower():
                print(premise)
                raise Exception(t)

        yield premise, hypothesis, tokens_premise, tokens_hypothesis


def load_nli_explain_1(name):
    path_idx = os.path.join(corpus_dir, "{}.txt".format(name))
    path_text = os.path.join(corpus_dir, "{}.csv".format(name))

    reader = csv.reader(open(path_text, "r", encoding="utf-8"), delimiter=',')

    texts_list = []
    for row in reader:
        premise = row[0]
        hypothesis = row[1]
        texts_list.append((premise, hypothesis))

    f = open(path_idx, "r")
    indice_list = []
    for line in f:
        p_indice, h_indice = line.split(",")
        p_indice = list([int(t) for t in p_indice.strip().split()])
        h_indice = list([int(t) for t in h_indice.strip().split()])
        indice_list.append((p_indice, h_indice))

    def complement(source, whole):
        return list(set(whole) - set(source))

    texts_list = texts_list[:len(indice_list)]
    debug = False
    for (prem, hypo), (p_indice, h_indice) in zip(texts_list, indice_list):
        p_tokens = prem.split()
        h_tokens = hypo.split()
        if debug:
            print(len(p_tokens), len(p_indice),len(h_tokens), len(h_indice))
            for idx in p_indice:
                print(p_tokens[idx], end=" ")
            print(" | ", end="")
            for idx in range(len(p_tokens)):
                if idx not in p_indice:
                    print(p_tokens[idx], end=" ")
            print("")
            for idx in h_indice:
                print(h_tokens[idx], end=" ")
            print(" | ", end="")
            for idx in range(len(h_tokens)):
                if idx not in h_indice:
                    print(h_tokens[idx], end=" ")
            print("")
        yield prem, hypo, p_indice, h_indice

def load_nli_explain_2(name_idx, name_text):
    path_idx = os.path.join(corpus_dir, "{}.csv".format(name_idx))
    path_text = os.path.join(corpus_dir, "{}.csv".format(name_text))

    reader = csv.reader(open(path_text, "r"), delimiter=',')

    texts_list = []
    for row in reader:
        premise = row[0]
        hypothesis = row[1]
        texts_list.append((premise, hypothesis))

    reader2 = csv.reader(open(path_idx, "r"), delimiter=",")
    indice_list = []
    for row in reader2:
        p_indice, h_indice = row[0], row[1]
        p_indice = list([int(t) for t in p_indice.strip().split()])
        h_indice = list([int(t) for t in h_indice.strip().split()])
        indice_list.append((p_indice, h_indice))

    texts_list = texts_list[:len(indice_list)]
    debug = False
    for (prem, hypo), (p_indice, h_indice) in zip(texts_list, indice_list):
        p_tokens = prem.split()
        h_tokens = hypo.split()
        if debug:
            print(len(p_tokens), len(p_indice),len(h_tokens), len(h_indice))
            for idx in p_indice:
                print(p_tokens[idx], end=" ")
            print(" | ", end="")
            for idx in range(len(p_tokens)):
                if idx not in p_indice:
                    print(p_tokens[idx], end=" ")
            print("")
            for idx in h_indice:
                print(h_tokens[idx], end=" ")
            print(" | ", end="")
            for idx in range(len(h_tokens)):
                if idx not in h_indice:
                    print(h_tokens[idx], end=" ")
            print("")
        yield prem, hypo, p_indice, h_indice


def load_nli_explain_3(name_idx, name_text):
    path_idx = os.path.join(corpus_dir, "{}.csv".format(name_idx))
    path_text = os.path.join(corpus_dir, "{}.csv".format(name_text))

    reader = csv.reader(open(path_text, "r", errors='ignore'), delimiter=',')

    texts_list = []
    for row in reader:
        premise = row[0]
        hypothesis = row[1]
        texts_list.append((premise, hypothesis))

    reader2 = csv.reader(open(path_idx, "r"), delimiter=",")
    indice_list = []
    for row in reader2:
        id = int(row[0])
        p_indice, h_indice = row[1], row[2]
        p_indice = list([int(t) for t in p_indice.strip().split()])
        h_indice = list([int(t) for t in h_indice.strip().split()])
        indice_list.append((p_indice, h_indice))

    texts_list = texts_list[:len(indice_list)]
    debug = False
    for (prem, hypo), (p_indice, h_indice) in zip(texts_list, indice_list):
        p_tokens = prem.split()
        h_tokens = hypo.split()
        if debug:
            print(len(p_tokens), len(p_indice),len(h_tokens), len(h_indice))
            for idx in p_indice:
                print(p_tokens[idx], end=" ")
            print(" | ", end="")
            for idx in range(len(p_tokens)):
                if idx not in p_indice:
                    print(p_tokens[idx], end=" ")
            print("")
            for idx in h_indice:
                print(h_tokens[idx], end=" ")
            print(" | ", end="")
            for idx in range(len(h_tokens)):
                if idx not in h_indice:
                    print(h_tokens[idx], end=" ")
            print("")
        yield prem, hypo, p_indice, h_indice


def read_gold_label(file_name):
    file_path = os.path.join(corpus_dir, file_name)

    reader2 = csv.reader(open(file_path, "r"), delimiter=",")
    indice_list = []
    for row in reader2:
        id = int(row[0])
        p_indice, h_indice = row[1], row[2]
        p_indice = list([int(t) for t in p_indice.strip().split()])
        h_indice = list([int(t) for t in h_indice.strip().split()])
        indice_list.append((p_indice, h_indice))
    return indice_list



def load_plain_text(file_name):
    path_text = os.path.join(corpus_dir, file_name)

    reader = csv.reader(open(path_text, "r", encoding="utf-8", errors='ignore'), delimiter=',')

    texts_list = []
    for row in reader:
        premise = row[0]
        hypothesis = row[1]
        texts_list.append((premise, hypothesis))

    return texts_list



def load_nli(path):
    label_list = ["entailment", "neutral", "contradiction", ]

    for idx, line in enumerate(open(path, "rb")):
        if idx == 0: continue  # skip header
        line = line.strip().decode("utf-8")
        split_line = line.split("\t")
        s1, s2 = split_line[8:10]
        l = label_list.index(split_line[-1])
        yield s1, s2, l


def load_mnli_explain_0():
    return load_from_pickle("mnli_explain")
    explation = load_nli_explain()
    dev_file = os.path.join(corpus_dir, "dev_matched.tsv")
    mnli_data = load_nli(dev_file)

    def find(prem, hypothesis):
        for datum in mnli_data:
            s1, s2, l = datum
            if prem == s1.strip() and hypothesis == s2.strip():
                return datum
        print("Not found")
        raise Exception(prem)

    def token_match(tokens1, tokens2):
        gold_indice = []
        for token in tokens1:
            matches = []
            alt_token = [token+".", token+",", token[0].upper() + token[1:]]
            for idx, t in enumerate(tokens2):
                if token == t:
                    matches.append(idx)
                elif t in alt_token:
                    matches.append(idx)

            if len(matches) == 1:
                gold_indice.append(matches[0])
            else:
                for idx, t in enumerate(tokens2):
                    print((idx, t), end =" ")
                print("")
                print(token)
                print(matches)
                print("Select indice: " , end="")
                user_written = input()
                gold_indice += [int(t) for t in user_written.split()]
        return gold_indice

    data = []
    for entry in explation:
        p, h, pe, he = entry

        datum = find(p.strip(),h.strip())
        s1, s2, l = datum

        s1_tokenize = s1.split()
        s2_tokenize = s2.split()

        e_indice_p = token_match(pe, s1_tokenize)
        e_indice_h = token_match(he, s2_tokenize)

        data.append({
            'p': s1,
            'p_tokens': s1_tokenize,
            'h': s2,
            'h_tokens': s2_tokenize,
            'y': l,
            'p_explain':e_indice_p,
            'h_explain':e_indice_h
        })
    save_to_pickle(data, "mnli_explain")
    return data



def reformat_mnli_explain_0():
    data = load_mnli_explain_0()

    text_list = []
    indice_list = []

    for entry in data:
        text_list.append((entry['p'], entry['h']))
        indice_list.append((entry['p_explain', 'h_explain']))


def tf_stat(data):
    def tokenize(sent):
        if sent[-1] == '.':
            sent = sent[:-1]
        tokens = sent.split()
        return list([t.lower() for t in tokens])

    tf_count = Counter()
    for s1, s2, l in data:
        for s in [s1, s2]:
            tokens = tokenize(s)
            tf_count.update(tokens)

    save_to_pickle(tf_count, "mnli_tf")


if __name__ == "__main__":
    dev_file = os.path.join(corpus_dir, "dev_matched.tsv")
    mnli_data = load_nli(dev_file)
    tf_stat(mnli_data)

