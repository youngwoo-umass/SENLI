from list_lib import right

data_idx = 0


# Return List[Score, BinaryLabel]
def get_score_label(pred, gold):
    score_label_pair = []
    all_pred_set = set([e for _, e in pred])
    for ge in gold:
        if ge not in all_pred_set:
            print("WARNING {} is not in {} -- line {}".format(ge, all_pred_set, data_idx))

    for score, e in pred:
        score_label_pair.append((score, e in gold))
    return score_label_pair


def get_all_score_label(explains, golds):
    score_list_h = []
    score_list_p = []

    global data_idx
    data_idx = 0
    for pred, gold in zip(explains, golds):
        pred_p, pred_h = pred
        gold_p, gold_h = gold
        data_idx += 1
        if gold_p:
            s1 = get_score_label(pred_p, gold_p)
            score_list_p.extend(s1)
        if gold_h:
            s2 = get_score_label(pred_h, gold_h)
            score_list_h.extend(s2)
    return score_list_p, score_list_h



def tune_cut(pred_list, gold_list):
    if len(pred_list) != len(gold_list):
        print("Warning")
        print("pred len={}".format(len(pred_list)))
        print("gold len={}".format(len(gold_list)))

    def maximize_f1(score_list):
        if not score_list:
            return -1, -1
        tp = 0
        pp = 0
        all_p = sum(right(score_list))

        best_f1 = -1
        best_cut = 0
        score_list.sort(key=lambda x :x[0], reverse=True)

        for score, label in score_list:
            pp += 1
            if label:
                tp += 1

            prec = tp / pp
            recall = tp / all_p
            f1 = 2* prec * recall / (prec + recall) if prec+recall > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_cut = score

        return best_cut, best_f1

    def maximize_prec(score_list):
        if not score_list:
            return -1, -1
        tp = 0
        pp = 0
        all_p = sum(right(score_list))

        best_prec = -1
        best_cut = 0
        score_list.sort(key=lambda x :x[0], reverse=True)

        for score, label in score_list:
            pp += 1
            if label:
                tp += 1

            prec = tp / (pp + 10)
            recall = tp / all_p
            if prec > best_prec:
                best_prec = prec
                best_cut = score

        return best_cut, best_prec



    def maximize_acc(score_list):
        if not score_list:
            return -1, -1
        tp = 0
        pp = 0
        n_total = len(score_list)
        all_p = sum(right(score_list))
        all_n = n_total - all_p
        best_acc = -1
        best_cut = 0
        score_list.sort(key=lambda x :x[0], reverse=True)

        for score, label in score_list:
            pp += 1
            if label:
                tp += 1

            fn = pp - tp

            tn = all_n - fn
            acc = (tp + tn) / n_total
            if acc > best_acc:
                best_acc = acc
                best_cut = score

        return best_cut, best_acc

    p_info, h_info = get_all_score_label(pred_list, gold_list)
    return maximize_acc(p_info), maximize_acc(h_info)
    #return maximize_prec(p_info), maximize_prec(h_info)
    #return maximize_f1(p_info), maximize_f1(h_info)


def get_pr(score_list, cut):
    tp = 0
    pp = 0
    all_p = sum(right(score_list))

    for score, label in score_list:
        if score >= cut:
            pp += 1
            if label :
                tp += 1

    prec = tp / pp if pp > 0 else 0
    recall = tp / all_p
    f1 = 2* prec * recall / (prec + recall) if prec+recall > 0 else 0
    return prec, recall, f1



def get_acc(score_list, cut):
    suc = 0
    total = len(score_list)

    for score, label in score_list:
        p = score >= cut

        if label == p:
            suc += 1

    return suc / total

