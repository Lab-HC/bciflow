
from bciflow.datasets import cbcic
from bciflow.modules.core.kfold import classification_kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe import logpower
from bciflow.modules.analysis.metric_functions import accuracy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from bciflow.modules.sf.ea import ea

d1 = cbcic(subject = 1, path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/CBCIC')
pre_folding = {'tf': (chebyshevII, {})}
pos_folding = {'tl': (ea(), {}),
               'fe': (logpower, {'flating': True}),
               'clf': (lda(), {})}
start_window=d1['events']['cue'][0]+0.5

results = classification_kfold(target=d1,
                start_window=start_window, 
                pre_folding=pre_folding, 
                pos_folding=pos_folding)

print(results)
print(accuracy(results))
print("\n")
