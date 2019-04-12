import preprocessing as prep
import pathlib

# feature_type = ['spectrogram', 'mel_spectrogram', 'mfcc_static', 'mfcc_delta']
# audio_preprocess = ['mid', 'side', 'left', 'right'] # L+R, L-R, L, R

n_feats = 100
dataset_name = f'numfeats{n_feats}'

db_root = 'E:/datasets/'
# db_root = '/media/linse/dados/zanco/TCC/datasets/'
# db_root = '/mnt/dsp/zanco/datasets/'
# db_root = '../../datasets/'
# db_root = '/media/zanco/DADOS/zanco/datasets/'


# version = '2018'
# version = '2019'
version = '2019c'


preprocessor = prep.DataPreprocessing(db_root=db_root,
                                      version=version,
                                      n_feats=n_feats, 
                                      dataset_name=dataset_name,
                                      dataset_folder=f'../saved_features{version}',
                                      audio_preprocess='mid',
                                      feature_type='mel_spectrogram')

audio_lenght, feature_shape = preprocessor.get_shapes()
print(f'audio_lenght, feature_shape: {audio_lenght, feature_shape}')

preprocessor.process(overwrite=True)
preprocessor.generate_fold_meta(overwrite=True)


