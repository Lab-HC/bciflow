import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from bciflow.datasets.CBCIC import cbcic
from bciflow.modules.core.kfold import kfold, windowing, apply_pre_folding
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe import logpower
from bciflow.modules.analysis.metric_functions import accuracy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from bciflow.modules.sf.ea import ea

d1 = cbcic(subject = 1)
d2 = cbcic(subject = 2)
d3 = cbcic(subject = 3)


pre_folding = {'tf': (chebyshevII, {})}
pos_folding = {'tl': (ea(), {'source': [d2]}),
               'fe': (logpower, {'flating': True}),
               'clf': (lda(), {})}
start_window=d1['events']['cue'][0]+0.5

d2 = windowing(target=d2, start_test_window=[start_window])
d2 = apply_pre_folding(target_dict=d2, start_test_window=[start_window], pre_folding=pre_folding)

results = kfold(target=d1, 
                start_window=start_window, 
                pre_folding=pre_folding, 
                pos_folding=pos_folding)

print(results)
print(accuracy(results))
print("\n")
