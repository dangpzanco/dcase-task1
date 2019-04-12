import pathlib
import predictor as pred

# Definitions for FeatureExtractor
n_feats = 100
audio_preprocess = 'mid'
feature_type = 'mel_spectrogram'

# Definitions for FeatureNormalizer
version = '2018'
zarr_root = pathlib.Path(f'../saved_features{version}')
dataset_name = f'numfeats{n_feats}'
zarr_group = f'{feature_type}/{audio_preprocess}/{dataset_name}'
fold_num = 0

# Audio file (or list)
audio_files = ['airport-barcelona-0-0-a.wav',
            'airport-barcelona-0-1-a.wav',
            'airport-barcelona-0-2-a.wav',
            'airport-barcelona-0-3-a.wav',
            'airport-barcelona-0-4-a.wav',
            'airport-barcelona-0-5-a.wav',
            'airport-barcelona-0-6-a.wav',
            'airport-barcelona-0-7-a.wav',
            'airport-barcelona-0-8-a.wav',
            'airport-barcelona-0-9-a.wav']

# Extract features
preprocessor = pred.FeatureExtractor(n_feats=n_feats,
                                     audio_preprocess=audio_preprocess,
                                     feature_type=feature_type)
features = preprocessor.process(audio_files)

# Fold and model list
fold_list = [0, 1, 2, 3, 4]
model_list = ['models_exp_augmentation/monaural_mid100_mixup_0.2_fold02018-11-13T18_39_30.699398.h5',
            'models_exp_augmentation/monaural_mid100_mixup_0.2_fold12018-11-13T21_00_50.589642.h5',
            'models_exp_augmentation/monaural_mid100_mixup_0.2_fold22018-11-13T22_18_05.513530.h5',
            'models_exp_augmentation/monaural_mid100_mixup_0.2_fold32018-11-13T23_22_19.095619.h5',
            'models_exp_augmentation/monaural_mid100_mixup_0.2_fold42018-11-14T02_45_53.091727.h5']

# fold_list = [1]
# model_list = ['models_exp_augmentation/monaural_mid100_mixup_0.2_fold12018-11-13T21_00_50.589642.h5']


# Predict features
predictor = pred.Predictor(zarr_root=zarr_root,
                           zarr_group=zarr_group, 
                           fold_list=fold_list, 
                           model_list=model_list)
predictions = predictor.predict(features)


print(predictions)
