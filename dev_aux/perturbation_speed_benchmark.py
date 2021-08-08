import time
import numpy as np
from dev.delete_trsfmr import get_seq_deleter
from dev.explain_trainer import generate_alt_runs
from dev.nli_common import load_nli_dataset


def main():
    batch_size = 16
    eval_batches, train_dataset = load_nli_dataset(batch_size, True, 300)
    sample_deleter = get_seq_deleter(0.5)
    compare_deletion_num = 20
    def cls_batch_to_input(cls_batch):
        x0, x1, y = cls_batch
        return x0, x1

    print("Start")
    for idx, cls_batch in enumerate(train_dataset):
        st = time.time()
        x0, x1 = cls_batch_to_input(cls_batch)
        probs = [0] * len(x0)
        perturbed_insts, instance_infos, deleted_mask_list \
            = generate_alt_runs(x0, x1, probs,
                                compare_deletion_num, sample_deleter)

        if idx == 1:
            break

        ed = time.time()
        print("{}".format(ed -st))

    return NotImplemented


if __name__ == "__main__":
    main()