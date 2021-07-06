import os
from collections import Counter

from cpath import output_path
from data_generator.text_encoder import SEP_ID

visualize_dir = os.path.join(output_path, "visualize")

def make_explain_sentence(result, out_name):
    # For entailment, filter out tokens that are same across sentences.
    # From ' tokens in premise' in premise , '
    sent_format = "'{}' in premise implies '{}'."

    def important_tokens(tokens, scores):
        cut = 100
        max_score = max(scores)
        min_score = min(scores)
        out_tokens = []
        for i, token in enumerate(tokens):
            r = int((scores[i] - min_score) * 255 / (max_score - min_score))
            if r > cut:
                r = cut
                out_tokens.append((i,token))
            else:
                r = 0
        return out_tokens


    def remove(src_tokens, exclude_list):
        return remove_i(enumerate(src_tokens), exclude_list)

    def remove_i(src_tokens, exclude_list):
        new_tokens = []
        exclude_set = set(exclude_list)
        for i, token in src_tokens:
            if token not in exclude_set:
                new_tokens.append((i, token))
        return new_tokens

    def common_words(tokens_a, tokens_b):
        return set(tokens_a).intersection(set(tokens_b))

    def make_sentence(tokens):
        prev_i = -3
        out_arr = []
        now_str = []
        for i, token in tokens:
            if i-1 != prev_i:
                if now_str:
                    out_arr.append(" ".join(now_str))
                    now_str = []

            now_str.append(token)
            prev_i = i

        if now_str:
            out_arr.append(" ".join(now_str))

        return ", ".join(out_arr)

    outputs = []
    for entry in result:
        pred_p, pred_h, prem, hypo = entry

        prem_reason = important_tokens(prem, pred_p)
        prem_hard_reason = remove_i(prem_reason, hypo)
        hypo_hard_reason = remove(hypo, common_words(prem, hypo))

        A = make_sentence(prem_hard_reason)
        B = make_sentence(hypo_hard_reason)
        reason = sent_format.format(A,B)

        print("Prem:", " ".join(prem))
        print("Hypo:", " ".join(hypo))
        print("Reason: ", reason)
        outputs.append((" ".join(prem), " ".join(hypo), reason))




def print_color_html(word, r):
    r = 255 -r
    bg_color = ("%02x" % r) + ("%02x" % r) + "ff"

    html = "<td bgcolor=\"#{}\">&nbsp;{}&nbsp;</td>".format(bg_color, word)
    #    html = "<td>&nbsp;{}&nbsp;</td>".format(word)
    return html

def print_color_html_2(word, raw_score):
    r = int(raw_score * 30)
    if r > 0:
        r = 255 - r
        bg_color = ("%02x" % r) + ("%02x" % r) + "ff"
    else:
        r = 255 + r
        bg_color = "ff" + ("%02x" % r) + ("%02x" % r)

    html = "<td bgcolor=\"#{}\">&nbsp;{}&nbsp;</td>".format(bg_color, word)
    #    html = "<td>&nbsp;{}&nbsp;</td>".format(word)
    return html

def visualize_nli(result, out_name):
    f = open_file(out_name)

    f.write("<html>")
    f.write("<body>")
    f.write("<div width=\"400\">")
    for entry in result:
        f.write("<div>\n")
        pred_p, pred_h, prem, hypo, pred, y = entry

        #max_score = max(max(pred_p), max(pred_h))
        #min_score = min(min(pred_p), min(pred_h))
        f.write("Pred : {} \n".format(pred))
        f.write("Gold : {}<br>\n".format(y))
        f.write("<tr>")
        for display_name, tokens, scores in [("Premise", prem, pred_p), ("Hypothesis", hypo, pred_h)]:
            f.write("<td><b>{}<b></td>\n".format(display_name))
            f.write("<table style=\"border:1px solid\">")

            max_score = max(scores)
            min_score = min(scores)
            cut = sorted(scores)[int(len(scores)*0.2)]
            if max_score == min_score:
                max_score = min_score + 3

            def normalize(score):
                return int((score - min_score) * 255 / (max_score - min_score))

            for i, token in enumerate(tokens):
                print("{}({}) ".format(token, scores[i]), end="")

                r = normalize(scores[i])
                if r > 100:
                    r = 100
                else:
                    r = 0
                #f.write(print_color_html_2(token, scores[i]))
                f.write(print_color_html(token, r))
            print()
            f.write("</tr>")
            f.write("</tr></table>")

        f.write("</tr>")

        f.write("</div><hr>")

    f.write("</div>")
    f.write("</body>")
    f.write("</html>")

def open_file(out_name):
    file_path = os.path.join(visualize_dir, "{}.html".format(out_name))
    return open(file_path, "w")



def visualize_sent_pair(result, out_name):
    f = open_file(out_name)

    f.write("<html>")
    f.write("<body>")
    f.write("<div width=\"400\">")
    for entry in result:
        f.write("<div>\n")
        ex1, ex2, tokens1, tokens2, pred, y = entry

        f.write("Pred : {} \n".format(pred))
        f.write("Gold : {}<br>\n".format(y))
        f.write("<tr>")
        for display_name, tokens, scores in [("Sent1", tokens1, ex1), ("Sent2", tokens2, ex2)]:
            f.write("<td><b>{}<b></td>\n".format(display_name))
            f.write("<table style=\"border:1px solid\">")

            max_score = max(scores)
            min_score = min(scores)
            cut = sorted(scores)[int(len(scores)*0.2)]
            if max_score == min_score:
                max_score = min_score + 3

            def normalize(score):
                return int((score - min_score) * 255 / (max_score - min_score))

            for i, token in enumerate(tokens):
                print("{}({}) ".format(token, scores[i]), end="")

                r = normalize(scores[i])
                if r > 160:
                    r = r - 50
                else:
                    r = 0
                #f.write(print_color_html_2(token, scores[i]))
                f.write(print_color_html(token, r))
            print()
            f.write("</tr>")
            f.write("</tr></table>")

        f.write("</tr>")

        f.write("</div><hr>")

    f.write("</div>")
    f.write("</body>")
    f.write("</html>")

def visualize_single(result, out_name):
    f = open_file(out_name)

    f.write("<html>")
    f.write("<body>")
    f.write("<div width=\"400\">")
    for entry in result:
        f.write("<div>\n")
        tokens, scores, pred, y = entry

        #max_score = max(max(pred_p), max(pred_h))
        #min_score = min(min(pred_p), min(pred_h))
        f.write("Pred : {} \n".format(pred))
        f.write("Gold : {}<br>\n".format(y))
        f.write("<tr>")
        f.write("<table style=\"border:1px solid\">")

        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            max_score = min_score + 3

        def normalize(score):
            return int((score - min_score) * 255 / (max_score - min_score))

        for i, token in enumerate(tokens):
            print("{}({}) ".format(token, scores[i]), end="")

            r = normalize(scores[i])
            if r > 100:
                r = 100
            else:
                r = 0
            #f.write(print_color_html_2(token, scores[i]))
            f.write(print_color_html(token, r))
        print()
        f.write("</tr>")
        f.write("</tr></table>")

        f.write("</tr>")

        f.write("</div><hr>")

    f.write("</div>")
    f.write("</body>")
    f.write("</html>")



def visual_stream(result):
    f = open("visual_stream.html", "w")
    f.write("<html>")
    f.write("<body>")
    f.write("<div width=\"400\">")
    for entry in result:
        f.write("<div>\n")
        pred_p, pred_h, prem, hypo, pred, y = entry

        #max_score = max(max(pred_p), max(pred_h))
        #min_score = min(min(pred_p), min(pred_h))
        f.write("Pred : {} \n".format(pred))
        f.write("Gold : {}<br>\n".format(y))
        f.write("<tr>")
        for display_name, tokens, scores in [("Premise", prem, pred_p), ("Hypothesis", hypo, pred_h)]:
            f.write("<td><b>{}<b></td>\n".format(display_name))
            f.write("<table style=\"border:1px solid\">")

            max_score = max(scores)
            min_score = min(scores)
            cut = sorted(scores)[int(len(scores)*0.2)]
            if max_score == min_score:
                max_score = min_score + 3

            def normalize(score):
                return int((score - min_score) * 255 / (max_score - min_score))

            for i, token in enumerate(tokens):
                print("{}({}) ".format(token, scores[i]), end="")

                r = normalize(scores[i])
                if r > 100:
                    r = 100
                else:
                    r = 0
                #f.write(print_color_html_2(token, scores[i]))
                f.write(print_color_html(token, r))
            print()
            f.write("</tr>")
            f.write("</tr></table>")

        f.write("</tr>")

        f.write("</div><hr>")

    f.write("</div>")
    f.write("</body>")
    f.write("</html>")



def word_stat(result, out_name):
    top_cnt = {
        'Premise': Counter(),
        'Hypothesis': Counter(),
    }
    for entry in result:
        pred_p, pred_h, prem, hypo, pred, y = entry

        #max_score = max(max(pred_p), max(pred_h))
        #min_score = min(min(pred_p), min(pred_h))
        if y == 2:
            for display_name, tokens, scores in [("Premise", prem, pred_p), ("Hypothesis", hypo, pred_h)]:
                best_score = -909
                best_token = None
                for i, token in enumerate(tokens):
                    if scores[i] > best_score:
                        best_score = scores[i]
                        best_token = token

                top_cnt[display_name][best_token] += 1


    for display_name in ["Premise", "Hypothesis"]:
        print(display_name)
        total = sum(top_cnt[display_name].values())
        for key, item in top_cnt[display_name].most_common(10):
            print("{}\t{}\t{}".format(key, item, item/total))


def split_with_input_ids(np_arr, input_ids):

    for i in range(len(input_ids)):
        if input_ids[i] == SEP_ID:
            idx_sep1 = i
            break

    p = np_arr[1:idx_sep1]
    for i in range(idx_sep1 + 1, len(input_ids)):
        if input_ids[i] == SEP_ID:
            idx_sep2 = i
    h = np_arr[idx_sep1 + 1:idx_sep2]
    return p, h


def visualize_pair_run(name, sess, tensor_sout, tensor_ex_logits, batches, feed_dict, encoder):
    result = []
    for batch in batches:
        x0, x1, x2, y = batch
        logits, conf_logit = sess.run([tensor_sout, tensor_ex_logits],
                                           feed_dict=feed_dict(batch))
        predictions = logits.argmax(axis=1)
        for idx in range(len(x0)):
            input_ids = x0[idx]
            ex1, ex2= split_with_input_ids(conf_logit[idx], input_ids)
            enc1, enc2 = split_with_input_ids(input_ids, input_ids)
            tokens1 = encoder.decode_list(enc1)
            tokens2 = encoder.decode_list(enc2)
            result.append((ex1, ex2, tokens1, tokens2, predictions[idx], y[idx]))
    visualize_sent_pair(result, name)