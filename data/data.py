# =============================================================================
# Data Base File
# Ziping Chen, USC
# =============================================================================

import os

if os.environ['DNC_DL_FRAMEWORK'] == 'torch':
    from data.torch_data import get_data_npz, get_data
elif os.environ['DNC_DL_FRAMEWORK'] == 'tf.keras':
    from data.tf_data import get_data_npz, get_data
else:
    raise Exception("framework not support!!!")