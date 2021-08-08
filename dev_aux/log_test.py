
import tensorflow as tf


def main():
    test_log_dir = 'logs'
    summary_writer = tf.summary.create_file_writer(test_log_dir)

    tag = "conflict"
    for i in range(100):
        with summary_writer.as_default():
            mean_ap = 0.01 * i
            tf.summary.scalar('MAP:ex_{}'.format(tag), mean_ap, step=i)



if __name__ == "__main__":
    main()