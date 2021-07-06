import tensorflow as tf

from attribution.deleter_trsfmr import *
from misc_lib import *
from tf_util.tf_logging import tf_logging
from trainer.np_modules import *


class ExplainTrainer:
    def __init__(self, forward_runs, action_score, sess, rl_loss, sout, ex_logits,
                 train_rl, input_rf_mask, batch2feed_dict, target_class_set, hparam, log2):
        self.forward_runs = forward_runs
        self.action_score = action_score
        self.compare_deletion_num = 20

        # Model Information
        self.sess = sess
        self.rl_loss = rl_loss
        self.sout = sout
        self.ex_logits = ex_logits
        self.train_rl = train_rl
        self.input_rf_mask = input_rf_mask

        self.batch2feed_dict = batch2feed_dict
        self.target_class_set = target_class_set
        self.hparam = hparam

        self.log2 = log2

        self.logit2tag = over_zero
        self.loss_window = MovingWindow(self.hparam.batch_size)

    def train_batch(self, batch, summary):
        def sample_size():
            prob = [(1 ,0.8), (2 ,0.2)]
            v = random.random()
            for n, p in prob:
                v -= p
                if v < 0:
                    return n
            return 1
    
        ## Step 1) Prepare deletion RUNS
        def generate_alt_runs(batch):
            logits, ex_logit = self.sess.run([self.sout, self.ex_logits
                                              ],
                                             feed_dict=self.batch2feed_dict(batch)
                                             )
            x0, x1, x2, y = batch
    
    
            pred = np.argmax(logits, axis=1)
            instance_infos = []
            new_batches = []
            deleted_mask_list = []
            tag_size_list = []
            for i in range(len(logits)):
                if pred[i] in self.target_class_set:
                    info = {}
                    info['init_logit'] = logits[i]
                    info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                    ex_tags = self.logit2tag(ex_logit[i])
                    tf_logging.debug("EX_Score : {}".format(numpy_print(ex_logit[i])))
                    tag_size = np.count_nonzero(ex_tags)
                    tag_size_list.append(tag_size)
                    if tag_size > 10:
                        tf_logging.debug("#Tagged token={}".format(tag_size))
    
                    info['idx_delete_tagged'] = len(new_batches)
                    new_batches.append(token_delete(ex_tags, x0[i], x1[i], x2[i]))
                    deleted_mask_list.append(ex_tags)
    
                    indice_delete_random = []
    
                    for _ in range(self.compare_deletion_num):
                        indice_delete_random.append(len(new_batches))
                        x_list, delete_mask = seq_delete_inner(sample_size(), x0[i], x1[i], x2[i])
                        new_batches.append(x_list)
                        deleted_mask_list.append(delete_mask)
    
                    info['indice_delete_random'] = indice_delete_random
                    instance_infos.append(info)
            if tag_size_list:
                avg_tag_size = average(tag_size_list)
                tf_logging.debug("avg Tagged token#={}".format(avg_tag_size))
            return new_batches, instance_infos, deleted_mask_list
    
        new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch)
    
        if not new_batches:
            self.log2.debug("Skip this batch")
            return
        ## Step 2) Execute deletion Runs
        alt_logits = self.forward_runs(new_batches)
    
        def reinforce_one(good_action, input_x):
            pos_reward_indice = np.int_(good_action)
            loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.1
            x0 ,x1 ,x2 ,y = input_x
            reward_payload = (x0, x1, x2, y, loss_mask)
            return reward_payload
    
        reinforce = reinforce_one
    
        ## Step 3) Calc reward
        def calc_reward(alt_logits, instance_infos, deleted_mask_list):
            models_score_list = []
            reinforce_payload_list = []
            num_tag_list = []
            pos_win = 0
            pos_trial = 0
            for info in instance_infos:
                init_output = info['init_logit']
                models_after_output = alt_logits[info['idx_delete_tagged']]
                input_x = info['orig_input']
    
                predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                num_tag = np.count_nonzero(predicted_action)
                num_tag_list.append(num_tag)
                models_score = self.action_score(init_output, models_after_output, predicted_action)
                models_score_list.append(models_score)

                good_action = predicted_action
                best_score = models_score
                for idx_delete_random in info['indice_delete_random']:
                    alt_after_output = alt_logits[idx_delete_random]
                    random_action = deleted_mask_list[idx_delete_random]
                    alt_score = self.action_score(init_output, alt_after_output, random_action)
                    if alt_score > best_score :
                        best_score = alt_score
                        good_action = random_action
    
                reward_payload = reinforce(good_action, input_x)
                reinforce_payload_list.append(reward_payload)
                if models_score >= best_score:
                    pos_win += 1
                pos_trial += 1
    
            match_rate = pos_win / pos_trial
            avg_score = average(models_score_list)
            self.log2.debug("drop score : {0:.4f}  suc_rate : {1:0.2f}".format(avg_score, match_rate))
            summary.value.add(tag='#Tags', simple_value=average(num_tag_list))
            summary.value.add(tag='Score', simple_value=avg_score)
            summary.value.add(tag='Success', simple_value=match_rate)
            return reinforce_payload_list
    
        reinforce_payload = calc_reward(alt_logits, instance_infos, deleted_mask_list)
    
        def commit_reward(reinforce_payload):
            batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
            rl_loss_list = []
            for batch in batches:
                x0, x1, x2, y, rf_mask = batch
                feed_dict = self.batch2feed_dict((x0,x1,x2,y))
                feed_dict[self.input_rf_mask] = rf_mask
                _, rl_loss, conf_logits, = self.sess.run([self.train_rl, self.rl_loss,
                                                          self.ex_logits,
                                                          ],
                                                         feed_dict=feed_dict)
                rl_loss_list.append((rl_loss, len(x0)))
            return rl_loss_list
        
        ## Step 4) Update gradient
        rl_loss_list = commit_reward(reinforce_payload)
        self.loss_window.append_list(rl_loss_list)

        window_rl_loss = self.loss_window.get_average()
        summary.value.add(tag='RL_Loss', simple_value=window_rl_loss)


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
        print("Using num_deletion : ", num_deletion)

        self.loss_window = list([MovingWindow(self.batch_size) for _ in range(self.num_tags)])
        self.sample_deleter = get_seq_deleter(self.g_val)

        self.action_to_label = action_to_label
        self.get_null_label = get_null_label

    @staticmethod
    def get_best_deletion(informative_score_fn, init_output, alt_logits, deleted_mask_list):
        good_action = None
        best_score = -1e5
        for after_logits, deleted_indices in zip(alt_logits, deleted_mask_list):
            alt_score = informative_score_fn(init_output, after_logits, deleted_indices)
            if alt_score > best_score:
                best_score = alt_score
                good_action = deleted_indices

        return good_action, best_score

    def calc_reward(self,
                    all_alt_logits,
                    instance_infos,
                    all_deleted_mask_list
                    ):
        models_score_list = []
        reinforce_payload_list = []
        num_tag_list = []
        for info in instance_infos:
            init_output = info['init_logit']
            input_x = info['orig_input']
            label_list = []
            for tag_idx, informative_score_fn in enumerate(self.informative_score_fn_list):
                alt_logits = [all_alt_logits[i] for i in info['indice_delete_random']]
                deleted_mask_list = [all_deleted_mask_list[i] for i in info['indice_delete_random']]
                good_action, best_score = self.get_best_deletion(informative_score_fn, init_output, alt_logits, deleted_mask_list)

                if best_score > self.drop_thres:
                    label = self.action_to_label(good_action)
                else:
                    label = self.get_null_label(good_action)
                label_list += label

            x0, x1, x2, y = input_x
            reward_payload = [x0, x1, x2, y] + label_list
            reinforce_payload_list.append(reward_payload)

        avg_score = average(models_score_list)
        summary = tf.Summary()
        summary.value.add(tag='#Tags', simple_value=average(num_tag_list))
        summary.value.add(tag='Score', simple_value=avg_score)
        return reinforce_payload_list, summary

    # train_fn : (batch) -> train_op execute, and loss return
    def train_batch(self, batch, train_fn):
        def sample_size():
            prob = [(1, 0.8), (2, 0.2)]
            v = random.random()
            for n, p in prob:
                v -= p
                if v < 0:
                    return n
            return 1
        #
        # def forward_run(batch):
        #     logits, = sess.run([self.logits], feed_dict=self.batch2feed_dict(batch))
        #     return logits

        def forward_runs(insts):
            batches = get_batches_ex(insts, self.batch_size, 3)
            alt_logits = []
            for b in batches:
                t = self.forward_run(b)
                alt_logits.append(t)
            alt_logits = np.concatenate(alt_logits)
            return alt_logits

        # Step 1) Prepare deletion RUNS
        def generate_alt_runs(batch):
            logits = self.forward_run(batch)
            x0, x1, x2, y = batch
            instance_infos = []
            new_insts = []
            deleted_mask_list = []
            tag_size_list = []
            for i in range(len(logits)):
                info = {}
                info['init_logit'] = logits[i]
                info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                indice_delete_random = []
                for _ in range(self.compare_deletion_num):
                    indice_delete_random.append(len(new_insts))
                    x_list, delete_mask = self.sample_deleter(sample_size(), x0[i], x1[i], x2[i])
                    new_insts.append(x_list)
                    deleted_mask_list.append(delete_mask)

                info['indice_delete_random'] = indice_delete_random
                instance_infos.append(info)
            if tag_size_list:
                avg_tag_size = average(tag_size_list)
                tf_logging.debug("avg Tagged token#={}".format(avg_tag_size))
            return new_insts, instance_infos, deleted_mask_list

        new_insts, instance_infos, deleted_mask_list = generate_alt_runs(batch)

        assert new_insts
        # Step 2) Execute deletion Runs
        alt_logits = forward_runs(new_insts)

        # Step 3) Calc reward
        def commit_reward(reinforce_payload):
            batches = get_batches_ex(reinforce_payload, self.batch_size, self.commit_input_len)
            ex_loss_list = []
            for batch in batches:
                ex_losses = train_fn(batch)
                x0 = batch[0]
                ex_loss_list.append((ex_losses, len(x0)))
            return ex_loss_list

        reinforce_payload, summary = self.calc_reward(alt_logits, instance_infos, deleted_mask_list)
        ## Step 4) Update gradient
        ex_loss_list = commit_reward(reinforce_payload)
        for ex_losses, num_insts in ex_loss_list:
            # ex_losses [num_tags]
            for tag_idx, loss_val in enumerate(ex_losses):
                self.loss_window[tag_idx].append(loss_val, num_insts)

        for tag_idx in range(self.num_tags):
            window_rl_loss = self.loss_window[tag_idx].get_average()
            summary.value.add(tag='Ex_loss_{}'.format(tag_idx), simple_value=window_rl_loss)

        return summary
