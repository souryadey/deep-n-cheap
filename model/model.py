# =============================================================================
# Model Base File
# Ziping Chen, USC
# =============================================================================

import os

if os.environ['DNC_DL_FRAMEWORK'] == 'torch':
    from model.torch_model import run_network, get_numparams, net_kws_defaults, run_kws_defaults, nn_activations
elif os.environ['DNC_DL_FRAMEWORK'] == 'tf.keras':
    from model.tf_model import run_network, get_numparams, net_kws_defaults, run_kws_defaults, nn_activations
else:
    raise Exception("framework not support!!!")