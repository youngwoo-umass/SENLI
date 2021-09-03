import numpy as np
import tensorflow as tf

from dev.delete_trsfmr import *
from misc_lib import *
from senli_log import senli_logging


# Step 1) Prepare deletion RUNS
def generate_alt_runs(x0, x1, probs,
                      compare_deletion_num,
                      sample_deleter,
                      ):
    if tf.is_tensor(x0):
        x0 = x0.numpy()
        x1 = x1.numpy()

    def draw_sample_size():
        prob = [(1, 0.8), (2, 0.2)]
        v = random.random()
        for n, p in prob:
            v -= p
            if v < 0:
                return n
        return 1
    senli_logging.debug("generate_alt_runs START")
    instance_infos = []
    new_insts = []
    deleted_mask_list = []
    tag_size_list = []
    for i in range(len(probs)):
        info = {}
        info['init_logit'] = probs[i]
        info['orig_input'] = (x0[i], x1[i])
        indice_delete_random = []
        for _ in range(compare_deletion_num):
            indice_delete_random.append(len(new_insts))
            x_list, delete_mask = sample_deleter(draw_sample_size(), x0[i], x1[i])
            new_insts.append(x_list)
            deleted_mask_list.append(delete_mask)
        info['indice_delete_random'] = indice_delete_random
        instance_infos.append(info)
    senli_logging.debug("generate_alt_runs : {} perturbed insts from {} insts".format(
        len(new_insts),
        len(probs),
    ))
    if tag_size_list:
        avg_tag_size = average(tag_size_list)
        senli_logging.debug("avg Tagged token#={}".format(avg_tag_size))
    return new_insts, instance_infos, deleted_mask_list


class ExplainTrainerM:
    def __init__(self,
                 informative_score_fn_list, #
                 num_tags,
                 action_to_label,
                 get_null_label,
                 forward_run,
                 batch_size,
                 num_deletion,
                 g_val,
                 drop_thres,
                 cls_batch_to_input
                 ):
        self.num_tags = num_tags

        self.informative_score_fn_list = informative_score_fn_list
        self.compare_deletion_num = num_deletion
        self.commit_input_len = 4 + num_tags * 2

        # Model Information
        self.drop_thres = drop_thres
        self.g_val = g_val
        self.batch_size = batch_size
        self.forward_run = forward_run
        self.sample_deleter = get_seq_deleter(self.g_val)

        self.action_to_label = action_to_label
        self.get_null_label = get_null_label
        self.forward_spec = (
            tf.TensorSpec(shape=(None, ), dtype=tf.int32),
            tf.TensorSpec(shape=(None, ), dtype=tf.int32),
        )
        self.ex_supervise_spec = [
            tf.TensorSpec(shape=(None, ), dtype=tf.int32),
            tf.TensorSpec(shape=(None, ), dtype=tf.int32),
        ]
        for _ in range(num_tags):
            self.ex_supervise_spec.extend(
                (tf.TensorSpec(shape=(None, ), dtype=tf.int32), tf.TensorSpec(shape=(1,), dtype=tf.int32))
                 )
        self.ex_supervise_spec = tuple(self.ex_supervise_spec)
        self.cls_batch_to_input = cls_batch_to_input

    @staticmethod
    def get_best_deletion(informative_score_fn, init_output, alt_logits, deleted_mask_list):
        good_action = None
        best_score: float = -1e5
        for after_logits, deleted_indices in zip(alt_logits, deleted_mask_list):
            assert type(after_logits) == np.ndarray
            alt_score = informative_score_fn(init_output, after_logits, deleted_indices)
            if alt_score > best_score:
                best_score = alt_score
                good_action = deleted_indices

        return good_action, best_score

    def calc_reward(self,
                    all_alt_logits,
                    instance_infos: List[Dict],
                    all_deleted_mask_list
                    ):
        reinforce_payload_list = []
        for info in instance_infos:
            init_output = info['init_logit']
            input_x = info['orig_input']
            label_list = []
            for tag_idx, informative_score_fn in enumerate(self.informative_score_fn_list):
                alt_logits = [all_alt_logits[i] for i in info['indice_delete_random']]
                deleted_mask_list = [all_deleted_mask_list[i] for i in info['indice_delete_random']]
                good_action, best_score = self.get_best_deletion(informative_score_fn, init_output, alt_logits, deleted_mask_list)

                if best_score > self.drop_thres:
                    label, mask = self.action_to_label(good_action)
                else:
                    label, mask = self.get_null_label(good_action)
                label_list += (np.array(label), np.array(mask))

            x0, x1 = input_x
            x0 = np.array(x0)
            x1 = np.array(x1)

            reward_payload = [x0, x1] + label_list
            reinforce_payload_list.append(tuple(reward_payload))

        return reinforce_payload_list

    def weak_supervision(self, cls_batch, ex_train_fn) -> np.ndarray:
        senli_logging.debug("weak_supervision::train_batch ENTRY")
        probs = self.forward_run([cls_batch])
        x0, x1 = self.cls_batch_to_input(cls_batch)
        perturbed_insts, instance_infos, deleted_mask_list\
            = generate_alt_runs(x0, x1, probs,
                                self.compare_deletion_num, self.sample_deleter)

        def inst_generator():
            for e in perturbed_insts:
                yield e
        senli_logging.debug("weak_supervision::train_batch tf.data.Dataset.from_generator")
        perturbation_batches = tf.data.Dataset.from_generator(
            inst_generator,
            output_signature=self.forward_spec
        ).batch(self.batch_size)

        # Step 2) Execute deletion Runs
        senli_logging.debug("weak_supervision::train_batch execute perturbed runs ")
        alt_logits = self.forward_run(perturbation_batches)
        # for b in perturbation_batches:
        #     t = self.forward_run(b)
        #     alt_logits.append(t)
        # alt_logits = np.concatenate(alt_logits)

        # Step 3) Calc reward
        # reinforce_payload : tf.Dataset
        def commit_reward(reinforce_payload: List) -> np.ndarray:
            def gen():
                yield from reinforce_payload
            ex_supervise_data = tf.data.Dataset.from_generator(gen, output_signature=self.ex_supervise_spec)
            ex_supervise_data = ex_supervise_data.batch(self.batch_size)
            return ex_train_fn(ex_supervise_data)

        senli_logging.debug("ExplainTrainerM:: calc reward ")
        reinforce_payload: List = self.calc_reward(alt_logits, instance_infos, deleted_mask_list)
        ## Step 4) Update gradient
        ex_losses: np.ndarray = commit_reward(reinforce_payload)
        return ex_losses
