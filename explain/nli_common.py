from misc_lib import average
from tf_util.tf_logging import tf_logging
from trainer.model_saver import save_model_to_dir_path


def save_fn_factory(sess, save_dir, global_step):
    return save_model_to_dir_path(sess, save_dir, global_step)


def train_classification_factory(sess, loss_tensor, acc_tensor, train_op, batch2feed_dict, batch, step_i):
    loss_val, acc, _ = sess.run([loss_tensor, acc_tensor, train_op,
                                                ],
                                               feed_dict=batch2feed_dict(batch)
                                               )
    tf_logging.debug("Step {0} train loss={1:.04f} acc={2:.04f}".format(step_i, loss_val, acc))
    return loss_val, acc


def train_debug_factory(sess, loss_tensor, acc_tensor, gradient, batch2feed_dict, batch, step_i):
    loss_val, acc, gradient = sess.run([loss_tensor, acc_tensor, gradient,
                                                ],
                                               feed_dict=batch2feed_dict(batch)
                                               )
    tf_logging.debug("Step {0} train loss={1:.04f} acc={2:.04f}".format(step_i, loss_val, acc))
    return loss_val, acc


def valid_fn_factory(sess, dev_batches, loss_tensor, acc_tensor, global_step_tensor, batch2feed_dict):
    loss_list = []
    acc_list = []
    for batch in dev_batches:
        loss_val, acc, g_step_val = sess.run([loss_tensor, acc_tensor, global_step_tensor],
                                               feed_dict=batch2feed_dict(batch)
                                               )
        loss_list.append(loss_val)
        acc_list.append(acc)
    tf_logging.info("Step dev step={0} loss={1:.04f} acc={2:.03f}".format(g_step_val, average(loss_list), average(acc_list)))

    return average(acc_list)