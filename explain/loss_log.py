import os

from cpath import output_path


def pretty_print(i):
    file_path = os.path.join(output_path, "report_log", "LossFn_{}_conflict".format(i))
    f = open(file_path, "r")
    train_loss = {}
    valid_info = {}
    for line in f:
        tokens = line.split()
        step = int(tokens[1])
        if tokens[0] == "Train_loss":
            train_loss[step] = float(tokens[2])
        elif tokens[0] == "Eval":
            valid_info[step] = float(tokens[2])

    file_path = os.path.join(output_path, "report_log", "train_loss_{}".format(i))
    f_t = open(file_path, "w")
    file_path = os.path.join(output_path, "report_log", "valid_p1_{}".format(i))
    f_v = open(file_path, "w")

    print(i)
    for step in range(0, 1000, 25):
        try:
            l = train_loss[step]
        except KeyError as e:
            l = train_loss[step-1]

        s = "\t({},{})\n".format(step, l)
        f_t.write(s)
        s = "\t({},{})\n".format(step, valid_info[step])
        f_v.write(s)
    f_t.close()
    f_v.close()



if __name__ == "__main__":
    for i in [2]:
        pretty_print(i)

