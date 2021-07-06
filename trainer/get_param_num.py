from tf_v2_support import tf1

def get_param_num():
    total_parameters = 0
    for variable in tf1.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    return total_parameters
