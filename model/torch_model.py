# =============================================================================
# Pytorch implementation of neural networks
# Sourya Dey, USC
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import pickle
from tqdm import tqdm
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
net_kws_defaults = {
                    'act': 'relu',
                    'out_channels': [],
                    'kernel_sizes': [3],
                    'paddings': [], #Fixed acc to kernel_size, i.e. 1 for k=3, 2 for k=5, etc
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
                'relu': F.relu,
                'tanh': F.tanh,
                'sigmoid': F.sigmoid,
                }

nn_activations = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'sigmoid': nn.Sigmoid
                 }

# =============================================================================
# Classes
# =============================================================================
class Net(nn.Module):
    def __init__(self, input_size = [3,32,32], output_size = 10, **kw):
        '''
        *** Create Pytorch net ***
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
        super().__init__()
        self.act = kw['act'] if 'act' in kw else net_kws_defaults['act']
        
        #### Conv ####
        self.out_channels = kw['out_channels'] if 'out_channels' in kw else net_kws_defaults['out_channels']
        self.num_layers_conv = len(self.out_channels)
        self.kernel_sizes = kw['kernel_sizes'] if 'kernel_sizes' in kw else self.num_layers_conv*net_kws_defaults['kernel_sizes']
        self.strides = kw['strides'] if 'strides' in kw else self.num_layers_conv*net_kws_defaults['strides']
        self.paddings = kw['paddings']  if 'paddings' in kw else [(ks-1)//2 for ks in self.kernel_sizes]
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
        self.conv = nn.ModuleDict({})
        for i in range(self.num_layers_conv):
            self.conv['conv-{0}'.format(i)] = nn.Conv2d(
                                    in_channels = input_size[0] if i==0 else self.out_channels[i-1],
                                    out_channels = self.out_channels[i],
                                    kernel_size = self.kernel_sizes[i],
                                    stride = self.strides[i],
                                    padding = self.paddings[i],
                                    dilation = self.dilations[i],
                                    groups = self.groups[i]
                                    )
                            
            
            if self.apply_maxpools[i] == 1:
                self.conv['mp-{0}'.format(i)] = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
            
            if self.apply_bns[i] == 1:
                self.conv['bn-{0}'.format(i)] = nn.BatchNorm2d(self.out_channels[i])

            self.conv['act-{0}'.format(i)] = nn_activations[self.act]()
            
            if self.apply_dropouts[i] == 1:
                self.conv['drop-{0}'.format(i)] = nn.Dropout(self.dropout_probs[dropout_index])
                dropout_index += 1
        
        if self.apply_gap == 1 and self.num_layers_conv > 0: #GAP is not done when there are no conv layers
            self.conv['gap'] = nn.AdaptiveAvgPool2d(output_size = 1) #this is basically global average pooling, i.e. input of (batch,cin,h,w) is converted to output (batch,cin,1,1)


        #### MLP ####
        self.mlp_input_size = self.get_mlp_input_size(input_size, self.conv)
        self.n_mlp = [self.mlp_input_size, output_size]
        if 'hidden_mlp' in kw:
            self.n_mlp[1:1] = kw['hidden_mlp'] #now n_mlp has the full MLP config, e.g. [800,100,10]
        self.num_hidden_layers_mlp = len(self.n_mlp[1:-1])
        self.apply_dropouts_mlp = kw['apply_dropouts_mlp'] if 'apply_dropouts_mlp' in kw else self.num_hidden_layers_mlp*net_kws_defaults['apply_dropouts_mlp']
        self.dropout_probs_mlp = kw['dropout_probs_mlp'] if 'dropout_probs_mlp' in kw else np.count_nonzero(self.apply_dropouts_mlp)*net_kws_defaults['dropout_probs_mlp']
        
        self.mlp = nn.ModuleList([])
        for i in range(len(self.n_mlp)-1):
            self.mlp.append(nn.Linear(self.n_mlp[i],self.n_mlp[i+1]))
        ## Do NOT put dropouts here instead, use F.dropout in forward()
    
    
    def get_mlp_input_size(self, input_size, prelayers):
        x = torch.ones(1,*input_size) #dummy input: all 1s, batch size 1
        with torch.no_grad():
            for layer in prelayers:
                x = prelayers[layer](x)
        return np.prod(x.size()[1:])
    
    
    def forward(self,x):
        rejoin = -1
        for layer in self.conv:
            if isinstance(self.conv[layer], nn.modules.conv.Conv2d):
                block = int(layer.split('-')[1])
            
                if block>0 and self.shortcuts[block-1] == 1:
                    rejoin = block+1
                    y = x.clone()
                    count_downsampling = sum(self.apply_maxpools[block:block+2]) + sum(self.strides[block:block+2]) - 2
                    for _ in range(count_downsampling):
                        y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)
                    y = F.pad(y, (0,0, 0,0, 0,self.out_channels[block+1] - self.out_channels[block-1], 0,0))
            if block==rejoin and 'act' in layer: #add shortcut to residual just before activation
                x += y
            
            x = self.conv[layer](x)
        
        x = x.view(-1, self.mlp_input_size) #flatten data to MLP inputs
        dropout_index = 0
        for i,layer in enumerate(self.mlp):
            x = layer(x)
            if i != len(self.mlp)-1: #last layer should not have regular activation
                x = F_activations[self.act](x)
                if self.apply_dropouts_mlp[i] == 1:
                    x = F.dropout(x, p=self.dropout_probs_mlp[dropout_index])
                    dropout_index += 1
        return x
    
def get_numparams(input_size, output_size, net_kw):
    ''' Get number of parameters in any net '''
    net = Net(input_size=input_size, output_size=output_size, **net_kw)
    numparams = sum([param.nelement() for param in net.parameters()])
    return numparams
    
class Hook():
    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)
    
    def hook_fn(self,layer,input,output):
        self.output = output
        
    def close(self):
        self.hook.remove()
# =============================================================================


# =============================================================================
# Helper methods
# =============================================================================
def train_batch(x,y, net, lossfunc, opt):
    '''
    Train on 1 batch of data
    '''
    opt.zero_grad()
    out = net(x)
    _, preds = torch.max(out, 1)
    correct = (preds==y).sum().item()
    loss = lossfunc(out,y)
    loss.backward()
    opt.step()   
    return correct, loss.item()


def eval_data(net, x, ensemble, **kw):
    '''
    *** General method to run only forward pass using net on data x ***
    kw:
        y : If given, find predictions and compute correct, else correct = None. Obviously y.shape[0] should be equal to x.shape[0]
        lossfunc : If given, compute loss, else loss = None
        hook : If desired, pass as a list of Hook(layer) objects
            E.g: hook = [Hook(layer) for layer in net.conv.layers if 'conv2d' in str(type(layer)).lower()]
            This finds intermediate outputs of all conv2d layers in the net
            These intermediate outputs are returned as raw_layer_outputs, and can be accessed as raw_layer_outputs[i].output
            Else raw_layer_outputs = None
    '''
    net.eval()
    with torch.no_grad():
        out = net(x)
        raw_layer_outputs = kw['hook'] if 'hook' in kw else None
        if 'y' in kw:
            _, preds = torch.max(out, 1)
            correct = (preds == kw['y']).sum().item()
            loss = kw['lossfunc'](out, kw['y']).item() if 'lossfunc' in kw else None
        else:
            correct = None
            loss = None
    return correct, loss, raw_layer_outputs, out if ensemble else None


def save_net(net = None, recs = None, filename = './results_new/new'):
    '''
    *** Saves net and records (if given) ***
    There are 2 ways to save a Pytorch net (see https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        1 - Save complete net: torch.save(net, filename.pt)
            Doing this needs a lot of space, e.g. >200MB for Reuters 100 epochs, hence not practical
            Loading is easy: net = torch.load(filename.pt)
        2 - Save state dict: torch.save(net.state_dict(), filename.pt)
            This needs LOT less space, e.g. 4 MB for Reuters 100 epochs
            Loading needs creation of the Net class with original kw:
                net = Net(args)
                net.load_state_dict(torch.load(filename.pt))
                SO STORE THE NET ARGS ALONG WITH NET, LIKE MAYBE IN A TEXT OR EXCEL FILE
    Use torch.load(map_location='cpu') to load CuDA models on local machine
    This message might appear when doing net.load_state_dict() - IncompatibleKeys(missing_keys=[], unexpected_keys=[]). IGNORE!
    '''
    if recs:
        with open(filename+'.pkl','wb') as f:
            pickle.dump(recs,f)
    if net:
        torch.save(net.state_dict(), filename+'.pt')
# =============================================================================



# =============================================================================
# Main method to run network
# =============================================================================
def run_network(
                data, input_size, output_size, problem_type, net_kw, run_kw,
                num_workers = 8, pin_memory = True,
                validate = True, val_patience = np.inf, test = False, ensemble = False,
                numepochs = 100,
                wt_init = nn.init.kaiming_normal_, bias_init = (lambda x : nn.init.constant_(x,0.1)),
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
    net = Net(input_size=input_size, output_size=output_size, **net_kw)
    
    ## Use GPUs if available ##
    if torch.cuda.device_count() > 1:
        print('Using {0} GPUs'.format(torch.cuda.device_count()))
        net = nn.DataParallel(net)    
    net.to(device) #convert parameters
    
    ## Initialize MLP params ##
    for i in range(len(net.mlp)):
        if wt_init is not None:
            wt_init(net.mlp[i].weight.data)
        if bias_init is not None:
            bias_init(net.mlp[i].bias.data)

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
        lossfunc = nn.CrossEntropyLoss(reduction='mean') ## IMPORTANT: By default, loss is AVERAGED across samples in a batch. If sum is desired, set reduction='sum'
    elif problem_type == 'regression':
        lossfunc = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(numepochs*milestone) for milestone in milestones], gamma=gamma)


# =============================================================================
# Data
# =============================================================================
    if type(data) == dict: #using Pytorch data loaders
        loader = True
        train_loader = torch.utils.data.DataLoader(data['train'], batch_size = batch_size, shuffle = True, num_workers=num_workers, pin_memory=pin_memory)
        if validate is True:
            val_loader = torch.utils.data.DataLoader(data['val'], batch_size = len(data['val']), num_workers=num_workers, pin_memory=pin_memory)
        if test is True:
            test_loader = torch.utils.data.DataLoader(data['test'], batch_size = len(data['test']), num_workers=num_workers, pin_memory=pin_memory)
    else: #using numpy
        loader = False
        xtr,ytr, xva,yva, xte,yte = data


# =============================================================================
#     Define records to collect
# =============================================================================
    recs = {
            'train_accs': np.zeros(numepochs), 'train_losses': np.zeros(numepochs),
            'val_accs': np.zeros(numepochs) if validate is True else None, 'val_losses': np.zeros(numepochs) if validate is True else None, 'val_final_outputs': numepochs*[0] #just initialize a dummy list
            #test_acc and test_loss are defined later
            }

    total_t = 0
    best_val_acc = -np.inf
    best_val_loss = np.inf
    
    for epoch in range(numepochs):
        if verbose:
            print('Epoch {0}'.format(epoch+1))
            
# =============================================================================
#         Run epoch
# =============================================================================
        ## Set up epoch ##
        numbatches = int(np.ceil(xtr.shape[0]/batch_size)) if not loader else len(train_loader)
        if not loader:
            shuff = torch.randperm(xtr.shape[0])		
            xtr, ytr = xtr[shuff], ytr[shuff]
        epoch_correct = 0
        epoch_loss = 0.
        
        ## Train ##
        t = time.time()
        net.train()
        for batch in tqdm(range(numbatches) if not loader else train_loader, leave=False):
            if not loader:
                inputs = xtr[batch*batch_size : (batch+1)*batch_size] #already converted to device
                labels = ytr[batch*batch_size : (batch+1)*batch_size]
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
            batch_correct, batch_loss = train_batch(x=inputs, y=labels, net=net, lossfunc=lossfunc, opt=opt)
            epoch_correct += batch_correct
            epoch_loss += batch_loss
            
        ## Time for epoch (don't collect for 1st epoch unless there is only 1 epoch) ##
        t_epoch = time.time() - t
        if epoch>0 or numepochs==1:
            total_t += t_epoch
            
        ## Save training records ##
        recs['train_accs'][epoch] = 100*epoch_correct/xtr.shape[0] if not loader else 100*epoch_correct/len(data['train'])
        recs['train_losses'][epoch] = epoch_loss/numbatches
        if verbose:
            print('Training Acc = {0}%, Loss = {1}'.format(np.round(recs['train_accs'][epoch],2), np.round(recs['train_losses'][epoch],3))) #put \n to make this appear on the next line after progress bar
        
# =============================================================================
#         Validate (optional)
# =============================================================================
        if validate is True:
            if not loader:
                correct, loss, _, final_outputs = eval_data(net=net, x=xva, ensemble=ensemble, y=yva, lossfunc=lossfunc)
                recs['val_accs'][epoch] = 100*correct/xva.shape[0]
                recs['val_losses'][epoch] = loss
            else:
                epoch_correct = 0
                epoch_loss = 0.
                for batch in tqdm(val_loader, leave=False):
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    batch_correct, batch_loss, _, final_outputs = eval_data(net=net, x=inputs, ensemble=ensemble, y=labels, lossfunc=lossfunc)
                    epoch_correct += batch_correct
                    epoch_loss += batch_loss
                val_acc = 100*epoch_correct/len(data['val'])
                val_loss = epoch_loss/len(val_loader)
                recs['val_accs'][epoch] = val_acc
                recs['val_losses'][epoch] = val_loss
            recs['val_final_outputs'][epoch] = final_outputs
            
            if verbose:
                print('Validation Acc = {0}%, Loss = {1}'.format(np.round(recs['val_accs'][epoch],2), np.round(recs['val_losses'][epoch],3)))  
            
# =============================================================================
#             Early stopping logic based on val_acc
# =============================================================================
            if problem_type == 'classification':
                if recs['val_accs'][epoch] > best_val_acc:
                    best_val_acc = recs['val_accs'][epoch]
                    best_val_ep = epoch+1
                    val_patience_counter = 0 #don't need to define this beforehand since this portion will always execute first when epoch==0
                else:
                    val_patience_counter += 1
                    if val_patience_counter == val_patience:
                        print('Early stopped after epoch {0}'.format(epoch+1))
                        numepochs = epoch+1 #effective numepochs after early stopping
                        break
            elif problem_type == 'regression':
                if recs['val_losses'][epoch] < best_val_loss:
                    best_val_loss = recs['val_losses'][epoch]
                    best_val_ep = epoch+1
                    val_patience_counter = 0 #don't need to define this beforehand since this portion will always execute first when epoch==0
                else:
                    val_patience_counter += 1
                    if val_patience_counter == val_patience:
                        print('Early stopped after epoch {0}'.format(epoch+1))
                        numepochs = epoch+1 #effective numepochs after early stopping
                        break
                
# =============================================================================
#         Schedule hyperparameters
# =============================================================================
        scheduler.step()
    
# =============================================================================
#     Final stuff at the end of training
# =============================================================================
    ## Final val metrics ##
    if validate is True:
        if problem_type == 'classification':
            print('\nBest validation accuracy = {0}% obtained in epoch {1}'.format(best_val_acc,best_val_ep))
        elif problem_type == 'regression':
            print('\nBest validation loss = {0} obtained in epoch {1}'.format(best_val_loss,best_val_ep))
    
    ## Testing ##
    if test is True:
        if not loader:
            correct, loss, _, final_outputs = eval_data(net=net, x=xte, ensemble=ensemble, y=yte, lossfunc=lossfunc)
            recs['test_acc'] = 100*correct/xte.shape[0]
            recs['test_loss'] = loss
        else:
            overall_correct = 0
            overall_loss = 0.
            for batch in tqdm(test_loader, leave=False):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                batch_correct, batch_loss, _, final_outputs = eval_data(net=net, x=inputs, ensemble=ensemble, y=labels, lossfunc=lossfunc)
                overall_correct += batch_correct
                overall_loss += batch_loss
            recs['test_acc'] = 100*overall_correct/len(data['test'])
            recs['test_loss'] = overall_loss/len(test_loader)
        recs['test_final_outputs'] = final_outputs
        print('Test accuracy = {0}%, Loss = {1}\n'.format(np.round(recs['test_acc'],2), np.round(recs['test_loss'],3)))
            
    ## Avg time taken per epoch ##
    recs['t_epoch'] = total_t/(numepochs-1) if numepochs>1 else total_t
    print('Avg time taken per epoch = {0}'.format(recs['t_epoch']))
    
    ## Cut recs as a result of early stopping ##
    recs = { **{key:recs[key][:numepochs] for key in recs if hasattr(recs[key],'__iter__')}, **{key:recs[key] for key in recs if not hasattr(recs[key],'__iter__')} } #this cuts the iterables like valaccs to the early stopping point, and keeps single values like testacc unchanged
    
    return net, recs
# =============================================================================

