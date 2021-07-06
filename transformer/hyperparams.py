
class HPTrain:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 16  # alias = N
    lr = 2e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    seq_max = 300 # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    hidden_units = 768  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_heads = 12
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    intermediate_size = 3072
    type_vocab_size = 2
    num_classes = 3

class HPEval:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 128
    lr = 2e-5
    logdir = 'logdir'

    # modele
    seq_max = 300 # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    hidden_units = 768  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_heads = 12
    dropout_rate = 0
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    intermediate_size = 3072
    type_vocab_size = 2
    g_val = 0.5
    num_classes = 3

