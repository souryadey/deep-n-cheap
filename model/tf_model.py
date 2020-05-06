# =============================================================================
# tf.keras implementation of neural networks
# Ziping Chen, USC
# =============================================================================

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
import time

# Ignore tf message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

net_kws_defaults = {
                    'act': 'relu',
                    'out_channels': [],
                    'kernel_sizes': [3],
                    'paddings': ['valid'],
                    'dilations': [1],
                    'groups': [1],
                    'strides': [1],
                    'apply_maxpools': [0],
                    'apply_gap': 1,
                    'apply_bns': [1],
                    'apply_dropouts': [1],
                    'dropout_probs': [0.1,0.3], #input layer, other layers
                    'shortcuts': [0],
                    'hidden_mlp': [],
                    'apply_dropouts_mlp': [1],
                    'dropout_probs_mlp': [0.2],
                    }

run_kws_defaults = {
                    'lr': 1e-3,
                    'gamma': 0.2,
                    'milestones': [0.5,0.75],
                    'weight_decay': 0.,
                    'batch_size': 256
                    }


F_activations = {
                'relu': tf.nn.relu,
                'tanh': tf.nn.tanh,
                'sigmoid': tf.nn.sigmoid
                }

nn_activations = {
                'relu': tf.keras.layers.Activation('relu'),
                'tanh': tf.keras.layers.Activation('tanh'),
                'sigmoid': tf.keras.layers.Activation('sigmoid'),
                }

class Net(tf.keras.Model):

    def __init__(self, input_size = [3,32,32], output_size = 10, problem_type = 'classification', **kw):
        '''
        *** Create tf.keras net ***
        input_size: Iterable. Size of 1 input. Example: [3,32,32] for CIFAR, [784] for MNIST
        output_size: Integer. #labels. Example: 100 for CIFAR-100, 39 for TIMIT phonemes
        kw:
            act: String. Activation for all layers. Must be pre-defined in F_activations and nn_activations. Default 'relu'
            --- CONV ---:
                    out_channels: Iterable. #filters in each conv layer, i.e. #conv layers. If no conv layer is needed, enter []          
                --- For the next kws, either pass an iterable of size = size of out_channels, OR leave blank to get default values ---
                        kernel_sizes: Default all 3
                        strides: Default all 1
                        paddings: Default values keep output size same as input for that kernel_size. Example 2 for kernel_size=5, 1 for kernel_size=3
                        dilations: Default all 1
                        groups: Default all 1
                        apply_bns: 1 to get BN layer after the current conv layer, else 0. Default all 1
                        apply_maxpools: 1 to get maxpool layer after the current conv layer, else 0. Default all 0
                        apply_dropouts: 1 to get dropout layer after the current conv layer, else 0. Default all 1
                        shortcuts: 1 to start shortcut after current conv layer, else 0. All shortcuts rejoin after 2 layers. Default all 0
                            2 consecutive elements of shortcuts cannot be 1, last 2 elements of shortcuts must be 0s
                            The shortcut portion has added 0s to compensate for channel increase, and avg pools to compensate for dwensampling
                    dropout_probs: Iterable of size = #1s in apply_dropouts. DROP probabilities for each dropout layer. Default first layer 0.1, all other 0.3
                        Eg: If apply_dropouts = [1,0,1,0], then dropout_probs = [0.1,0.3]. If apply_dropouts = [0,1,1,1], then dropout_probs = [0.3,0.3,0.3]
                    apply_gap: 1 to apply global average pooling just before MLPs, else 0. Default 1
            --- MLP ---:
                    hidden_mlp: Iterable. #nodes in the hidden layers only.
                    apply_dropouts_mlp: Whether to apply dropout after current hidden layer. Iterable of size = number of hidden layers. Default all 0
                    dropout_probs_mlp: As in dropout_probs for conv. Default all 0.5
                    
                    Examples:
                        If input_size=800, output_size=10, and hidden_mlp is not given, or is [], then the config will be [800,10]. By default, apply_dropouts_mlp = [], dropout_probs_mlp = []
                        If input_size=800, output_size=10, and hidden_mlp is [100,100], then the config will be [800,100,100,10]. apply_dropouts_mlp for example can be [1,0], then dropout_probs_mlp = [0.5] by default
        '''
        super(Net, self).__init__()
        self.act = kw['act'] if 'act' in kw else net_kws_defaults['act']
        #self.input = tf.keras.layers.Input(shape=input_size)   

        #### Conv ####
        self.out_channels = kw['out_channels'] if 'out_channels' in kw else net_kws_defaults['out_channels']
        self.num_layers_conv = len(self.out_channels)
        self.kernel_sizes = kw['kernel_sizes'] if 'kernel_sizes' in kw else self.num_layers_conv*net_kws_defaults['kernel_sizes']
        self.strides = kw['strides'] if 'strides' in kw else self.num_layers_conv*net_kws_defaults['strides']
        self.paddings = kw['paddings']  if 'paddings' in kw else self.num_layers_conv*['same']
        self.dilations = kw['dilations'] if 'dilations' in kw else self.num_layers_conv*net_kws_defaults['dilations']
        self.groups = kw['groups'] if 'groups' in kw else self.num_layers_conv*net_kws_defaults['groups']
        self.apply_bns = kw['apply_bns'] if 'apply_bns' in kw else self.num_layers_conv*net_kws_defaults['apply_bns']
        self.apply_maxpools = kw['apply_maxpools'] if 'apply_maxpools' in kw else self.num_layers_conv*net_kws_defaults['apply_maxpools']
        self.apply_gap = kw['apply_gap'] if 'apply_gap' in kw else net_kws_defaults['apply_gap']

        self.apply_dropouts = kw['apply_dropouts'] if 'apply_dropouts' in kw else self.num_layers_conv*net_kws_defaults['apply_dropouts']
        if 'dropout_probs' in kw:
            self.dropout_probs = kw['dropout_probs']
        else:
            self.dropout_probs = np.count_nonzero(self.apply_dropouts)*[net_kws_defaults['dropout_probs'][1]]
            if len(self.apply_dropouts)!=0 and self.apply_dropouts[0]==1:
                self.dropout_probs[0] = net_kws_defaults['dropout_probs'][0]

        self.shortcuts = kw['shortcuts'] if 'shortcuts' in kw else self.num_layers_conv*net_kws_defaults['shortcuts']

        dropout_index = 0

        self.conv = {}
        for i in range(self.num_layers_conv):
            self.conv['conv-{0}'.format(i)] = tf.keras.layers.Conv2D(
                                                filters=self.out_channels[i],
                                                kernel_size=self.kernel_sizes[i],
                                                strides=self.strides[i],
                                                padding=self.paddings[i], # NOTE: one of "valid" or "same"
                                                dilation_rate=self.dilations[i],
                                                # TODO: group conv
                                                )
            
            if self.apply_maxpools[i] == 1:
                self.conv['mp-{0}'.format(i)] = tf.keras.layers.MaxPool2D(
                                                    pool_size=2, # TODO: hyper-parameter for searching
                                                    )
            
            if self.apply_bns[i] == 1:
                self.conv['bn-{0}'.format(i)] = tf.keras.layers.BatchNormalization()

            self.conv['act-{0}'.format(i)] = nn_activations[self.act]

            if self.apply_dropouts[i] == 1:
                self.conv['drop-{0}'.format(i)] = tf.keras.layers.Dropout(self.dropout_probs[dropout_index])
                dropout_index += 1

        if self.apply_gap == 1 and self.num_layers_conv > 0: #GAP is not done when there are no conv layers
            self.conv['gap'] = tf.keras.layers.GlobalAveragePooling2D()

        #### MLP ####
        # self.mlp_input_size = self.get_mlp_input_size(input_size, self.conv)
        self.n_mlp = [-1, output_size] # tf.keras don't need input size
        if 'hidden_mlp' in kw:
            self.n_mlp[1:1] = kw['hidden_mlp'] #now n_mlp has the full MLP config, e.g. [-1,100,10]
        else:
            self.n_mlp[1:1] = net_kws_defaults['hidden_mlp']
        self.num_hidden_layers_mlp = len(self.n_mlp[1:-1])
        self.apply_dropouts_mlp = kw['apply_dropouts_mlp'] if 'apply_dropouts_mlp' in kw else self.num_hidden_layers_mlp*net_kws_defaults['apply_dropouts_mlp']
        self.dropout_probs_mlp = kw['dropout_probs_mlp'] if 'dropout_probs_mlp' in kw else np.count_nonzero(self.apply_dropouts_mlp)*net_kws_defaults['dropout_probs_mlp']
        self.mlp = {}
        dropout_index = 0
        for i in range(1, len(self.n_mlp)):
            if i != len(self.n_mlp) - 1:
                self.mlp['dense-{0}'.format(i - 1)] = tf.keras.layers.Dense(self.n_mlp[i], activation=F_activations[self.act])
                if self.apply_dropouts_mlp[i - 1] == 1:
                    self.mlp['drop-{0}'.format(i - 1)] = tf.keras.layers.Dropout(self.dropout_probs_mlp[dropout_index])
                    dropout_index += 1
            else:
                if problem_type == 'classification':
                    self.mlp['dense-{0}'.format(i - 1)] = tf.keras.layers.Dense(self.n_mlp[i], activation='softmax')
                elif problem_type == 'regression':
                    self.mlp['dense-{0}'.format(i - 1)] = tf.keras.layers.Dense(self.n_mlp[i])

    def call(self, inputs):
        rejoin = -1
        x = inputs
        for layer in self.conv:
            if isinstance(self.conv[layer], tf.keras.layers.Conv2D):
                block = int(layer.split('-')[1])
                if block>0 and self.shortcuts[block-1] == 1:
                    rejoin = block+1
                    y = x
                    count_downsampling = sum(self.apply_maxpools[block:block+2]) + sum(self.strides[block:block+2]) - 2
                    for _ in range(count_downsampling):
                        y = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(y)

                    y = tf.pad(y,[[0,0],[0,0],[0,0],[0,self.out_channels[block+1] - self.out_channels[block-1]]], 'CONSTANT')
            if block==rejoin and 'act' in layer: #add shortcut to residual just before activation
                x = tf.keras.layers.Add()([x, y])
            x = self.conv[layer](x)
        x = tf.keras.layers.Flatten()(x)
        for layer in self.mlp:
            x = self.mlp[layer](x)
        return x


def get_numparams(input_size, output_size, net_kw):
    ''' Get number of parameters in any net '''
    net = Net(input_size=input_size, output_size=output_size, **net_kw)
    # from NCHW to NHWC
    net.build(tuple([None] + input_size[1:] + [input_size[0]]))
    # net.trainable = True
    # numparams = sum([param.nelement() for param in net.parameters()])
    trainable_count = np.sum([K.count_params(w) for w in net.trainable_weights])
    return trainable_count

# =============================================================================
# Main method to run network
# =============================================================================
def run_network(
                data, input_size, output_size, problem_type, net_kw, run_kw,
                num_workers = 8, pin_memory = True,
                validate = True, val_patience = np.inf, test = False, ensemble = False,
                numepochs = 100,
                wt_init = None,#nn.init.kaiming_normal_, 
                bias_init = None,#(lambda x : nn.init.constant_(x,0.1)),
                verbose = True
               ):
    '''
    ARGS:
        data:
            6-ary tuple (xtr,ytr, xva,yva, xte,yte) from get_data_mlp(), OR
            Dict with keys 'train', 'val', 'test' from get_data_cnn()
        input_size, output_size, net_kw : See Net()
        run_kw:
            lr: Initial learning rate
            gamma: Learning rate decay coefficient
            milestones: When to step decay learning rate, e.g. 0.5 will decay lr halfway through training
            weight_decay: Default 0
            batch_size: Default 256
        num_workers, pin_memory: Only required if using Pytorch data loaders
            Generally, set num_workers equal to number of threads (e.g. my Macbook pro has 4 cores x 2 = 8 threads)
        validate: Whether to do validation at the end of every epoch.
        val_patience: If best val acc doesn't increase for this many epochs, then stop training. Set as np.inf to never stop training (until numepochs)
        test: True - Test at end, False - don't
        ensemble: If True, return feedforward soft outputs to be later used for ensembling
        numepochs: Self explanatory
        wt_init, bias_init: Respective pytorch functions
        verbose: Print messages
    
    RETURNS:
        net: Complete net
        recs: Dictionary with a key for each stat collected and corresponding value for all values of the stat
    '''
# =============================================================================
#     Create net
# =============================================================================
    net = Net(input_size=input_size, output_size=output_size, problem_type=problem_type, **net_kw)
    net.build(tuple([None] + input_size[1:] + [input_size[0]]))
    ## Use GPUs if available ##
    
    ## Initialize MLP params ##
    # TODO
    '''
    for i in range(len(net.mlp)):
        if wt_init is not None:
            wt_init(net.mlp[i].weight.data)
        if bias_init is not None:
            bias_init(net.mlp[i].bias.data)
    '''

# =============================================================================
#     Hyperparameters for the run
# =============================================================================
    lr = run_kw['lr'] if 'lr' in run_kw else run_kws_defaults['lr']
    gamma = run_kw['gamma'] if 'gamma' in run_kw else run_kws_defaults['gamma'] #previously used value according to decay = 1e-5 in keras = 0.9978 for ExponentialLR
    milestones = run_kw['milestones'] if 'milestones' in run_kw else run_kws_defaults['milestones']
    weight_decay = run_kw['weight_decay'] if 'weight_decay' in run_kw else run_kws_defaults['weight_decay']
    batch_size = run_kw['batch_size'] if 'batch_size' in run_kw else run_kws_defaults['batch_size']
    if not isinstance(batch_size,int):
        batch_size = batch_size.item() #this is required for pytorch
    
    if problem_type == 'classification':
        lossfunc = tf.keras.losses.SparseCategoricalCrossentropy()
    elif problem_type == 'regression':
        lossfunc = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=lr, decay=weight_decay)
    net.compile(optimizer=opt, loss=lossfunc, metrics=['accuracy'])
    trainable_count = np.sum([K.count_params(w) for w in net.trainable_weights])

    def multi_step_lr(epoch):
        LR_START = lr
        GAMMA = gamma
        NUMEPOCHS = numepochs
        step = 0
        for milestone in milestones:
            if epoch >= int(milestone * NUMEPOCHS):
                step += 1
        return LR_START * (GAMMA ** step)
        
    lr_callback = tf.keras.callbacks.LearningRateScheduler(multi_step_lr, verbose=verbose)

# =============================================================================
# Data
# =============================================================================
    xtr,ytr, xva,yva, xte,yte = data

# =============================================================================
#     Define records to collect
# =============================================================================
    recs = {}

    total_t = 0
    best_val_acc = -np.inf
    best_val_loss = np.inf
    
    class TimeHistory(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []
            # self.test_times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

        # def on_test_begin(self, batch, logs={}):
        #     self.test_time_start = time.time()

        # def on_test_end(self, batch, logs={}):
        #     self.test_times.append(time.time() - self.test_time_start)

    th_callback = TimeHistory()#verbose=verbose)
# =============================================================================
#         Run epoch
# =============================================================================
    if validate is True:
        if problem_type == 'classification':
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=val_patience, verbose=verbose, restore_best_weights=True)
        elif problem_type == 'regression':
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=val_patience, verbose=verbose, restore_best_weights=True)
        history = net.fit(
                x=xtr,
                y=ytr,
                verbose=verbose,
                validation_data=(xva, yva),
                batch_size=batch_size,
                epochs=numepochs,
                shuffle=True,
                use_multiprocessing=False,
                callbacks=[lr_callback, th_callback, es_callback])
        recs['val_accs'] = np.array(history.history['val_accuracy']) * 100
        recs['val_losses'] = history.history['val_loss']
        # recs['val_final_outputs'] = None # NOT use
    else:
        history = net.fit(
                x=xtr,
                y=ytr,
                verbose=verbose,
                batch_size=batch_size,
                epochs=numepochs,
                shuffle=True,
                use_multiprocessing=False,
                callbacks=[lr_callback, th_callback])
    recs['train_accs'] = np.array(history.history['accuracy']) * 100
    recs['train_losses'] = history.history['loss']

    total_t += np.sum(th_callback.times)

    ## Final val metrics ##
    if validate is True:
        if problem_type == 'classification':
            print('\nBest validation accuracy = {0}% obtained in epoch {1}'.format(np.max(recs['val_accs']), np.argmax(recs['val_accs']) + 1))
        elif problem_type == 'regression':
            print('\nBest validation loss = {0} obtained in epoch {1}'.format(np.min(recs['val_losses']), np.argmin(recs['val_losses']) + 1))

    if test is True:
        ret = net.evaluate(
                x=xte,
                y=yte,
                verbose=verbose,
                batch_size=batch_size,
                workers=1,
                use_multiprocessing=False)
        recs['test_acc'] = ret[1] * 100
        recs['test_loss'] = ret[0]
        # recs['test_final_outputs'] = None # NOT use
        print('Test accuracy = {0}%, Loss = {1}\n'.format(np.round(recs['test_acc'],2), np.round(recs['test_loss'],3)))
            
    ## Avg time taken per epoch ##
    recs['t_epoch'] = total_t/(numepochs-1) if numepochs>1 else total_t
    print('Avg time taken per epoch = {0}'.format(recs['t_epoch']))
    
    '''
    ## Cut recs as a result of early stopping ##
    recs = { **{key:recs[key][:numepochs] for key in recs if hasattr(recs[key],'__iter__')}, **{key:recs[key] for key in recs if not hasattr(recs[key],'__iter__')} } #this cuts the iterables like valaccs to the early stopping point, and keeps single values like testacc unchanged
    '''

    return net, recs
# =============================================================================
