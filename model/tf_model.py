import tensorflow as tf
import numpy as np

net_kws_defaults = {
                    'act': 'relu',
                    'out_channels': [1],
                    'kernel_sizes': [3],
                    'paddings': ['valid'], #Fixed acc to kernel_size, i.e. 1 for k=3, 2 for k=5, etc
                    'dilations': [1],
                    'groups': [1],
                    'strides': [1],
                    'apply_maxpools': [0],
                    'apply_gap': 1,
                    'apply_bns': [1],
                    'apply_dropouts': [1],
                    'dropout_probs': [0.1,0.3], #input layer, other layers
                    'shortcuts': [0],
                    'hidden_mlp': [1],
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
        'softmax': tf.nn.softmax,
        }
nn_activations = {
        'relu': tf.keras.layers.ReLU
        }

class Net(tf.keras.Model):

    def __init__(self, input_size = [3,32,32], output_size = 10, **kw):
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
        # print(self.num_layers_conv)
        self.kernel_sizes = kw['kernel_sizes'] if 'kernel_sizes' in kw else self.num_layers_conv*net_kws_defaults['kernel_sizes']
        self.strides = kw['strides'] if 'strides' in kw else self.num_layers_conv*net_kws_defaults['strides']
        self.paddings = kw['paddings']  if 'paddings' in kw else self.num_layers_conv*net_kws_defaults['paddings']#[(ks-1)//2 for ks in self.kernel_sizes]
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
            
            self.conv['act-{0}'.format(i)] = nn_activations[self.act]()

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
            self.mlp['dense-{0}'.format(i - 1)] = tf.keras.layers.Dense(self.n_mlp[i], activation=F_activations[self.act] if i != len(self.n_mlp) - 1 else F_activations['softmax'])
            if i != len(self.n_mlp) - 1 and self.apply_dropouts_mlp[i - 1] == 1:
                self.mlp['drop-{0}'.format(i - 1)] = tf.keras.layers.Dropout(self.dropout_probs_mlp[dropout_index])
                dropout_index += 1

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
                        y = tf.nn.avg_pool2d(y, ksize=2)
                    # TODO
                    assert False, "No Implementation"
                    y = tf.keras.layers.ZeroPadding2D()(y)
            if block==rejoin and 'act' in layer: #add shortcut to residual just before activation
                x = tf.keras.layers.Add()([x, y])
            x = self.conv[layer](x)
        x = tf.keras.layers.Flatten()(x)
        dropout_index = 0
        for layer in self.mlp:
            x = self.mlp[layer](x)

        return x
