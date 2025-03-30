import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from bciflow.datasets.cbcic import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe import logpower
from bciflow.modules.analysis.metric_functions import accuracy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

d1 = cbcic(subject = 1)
d2 = cbcic(subject = 2)

pre_folding = {'tf': (chebyshevII, {})}
pos_folding = {'fe': (logpower, {'flating': True}),
               'clf': (lda(), {})}

results1 = kfold(target=d1, 
                start_window=d1['events']['cue'][0]+0.5, 
                pre_folding=pre_folding, 
                pos_folding=pos_folding)

results2 = kfold(target=d2, 
                start_window=d2['events']['cue'][0]+0.5, 
                pre_folding=pre_folding, 
                pos_folding=pos_folding)

results3 = kfold(target=d1, 
                start_window=d1['events']['cue'][0]+0.5, 
                pre_folding=pre_folding, 
                pos_folding=pos_folding,
                source = [d2])

print(results1)
print(accuracy(results1))
print("\n")
print(results2)
print(accuracy(results2))
print("\n")
print(results3)
print(accuracy(results3))
print("\n")