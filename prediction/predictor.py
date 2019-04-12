
# Standard libraries
import pathlib

# Scientific stack
import numpy as np

# Chunked data
import zarr

# Audio processing
import dcase_util as du

# Tensorflow config
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
import keras



class FeatureExtractor():

    def __init__(self, fs=48000, win_length_seconds=40e-3, hop_length_seconds=20e-3, n_feats=100, n_fft=2048, window_type='hann', 
                 feature_type='mel_spectrogram', audio_preprocess='mid', minmax=False, audio_duration=10.0, **kwargs):
        super().__init__()

        self.__dict__.update(kwargs)
        
        self.fs = fs

        self.audio_preprocess = audio_preprocess
        self.minmax = minmax

        self.audio_length = int(audio_duration * self.fs)
        
        # Feature extractor input arguments
        args = dict(fs=fs, win_length_seconds=win_length_seconds, hop_length_seconds=hop_length_seconds, 
                n_mels=n_feats, n_mfcc=n_feats, n_dim=n_feats, spectrogram_type='magnitude', n_fft=n_fft, window_type=window_type, **kwargs)
        self.set_extractor(feature_type, args)

    def set_extractor(self, feature_type, args):
        self.feature_type = feature_type

        # Create feature extractor
        if feature_type == 'spectrogram':
            self.extractor = du.features.SpectralFeatureExtractor(**args)
        elif feature_type == 'mel_spectrogram':
            self.extractor = du.features.MelExtractor(**args)
        elif feature_type == 'mfcc_static':
            args['n_mels'] = int(n_fft/4)
            self.extractor = du.features.MfccStaticExtractor(**args)
        elif feature_type == 'mfcc_delta':
            args['n_mels'] = int(n_fft/4)
            self.extractor = du.features.MfccDeltaExtractor(**args)
        else:
            message = f'set_extractor(): Unknown feature type [{feature_type}]'
            raise ValueError(message)

    def process(self, audio_handle):

        feat_shape = self.get_shape()
        
        if type(audio_handle) is list:

            num_files = len(audio_handle)

            features = np.empty([num_files, *feat_shape])

            for i, file in enumerate(audio_handle):
                features[i,] = self._single_process(file)

        else:
            features = np.expand_dims(self._single_process(audio_handle), axis=0)

        return features

    def _single_process(self, audio_handle):

        if type(audio_handle) not in [np.array, du.containers.AudioContainer]:
            # Open audio file
            audio_container = du.containers.AudioContainer().load(
                filename=str(audio_handle),
                mono=False,
                fs=self.fs
            )
            audio_data = audio_container.data

        # Convert audio to mono (left, right, mid or side)
        audio_data = self.audio_mixing(audio_data, self.audio_preprocess, self.minmax)

        # Fix audio duration
        audio_data = self.fix_audio_length(audio_data, self.audio_length)

        # Extract features from the audio data
        features = self.feature_extraction(audio_data)

        return features



    def fix_audio_length(self, audio_data, length):

        current_size = audio_data.size

        if current_size == length:
            return audio_data
        elif current_size < length:
            new_audio_data = np.empty(length)
            new_audio_data[:current_size] = audio_data.copy()
            new_audio_data[current_size:] = 0
            return new_audio_data
        else:
            new_audio_data = audio_data[:length].copy()
            return new_audio_data

    def audio_mixing(self, audio_data, audio_preprocess, minmax):

        if len(audio_data.shape) == 2:
            # Audio mixing
            if audio_preprocess == 'left':
                audio_data = audio_data[0,:]
            elif audio_preprocess == 'right':
                audio_data = audio_data[1,:]
            elif audio_preprocess == 'mid':
                audio_data = audio_data[0,:] + audio_data[1,:]
            elif audio_preprocess == 'side':
                audio_data = audio_data[0,:] - audio_data[1,:]
            else:
                message = f'audio_mixing(): Unknown audio_preprocess type [{audio_preprocess}]'
                raise ValueError(message)

        # Optional normalization per audio snippet
        if minmax:
            audio_data /= np.maximum(audio_data.max(),-audio_data.min())

        return audio_data

    def feature_extraction(self, audio_data):
        return self.extractor.extract(audio_data)

    def get_shape(self):

        # Temporary array
        a = np.empty(self.audio_length)

        # Get feature vector shape
        feature_shape = self.extractor.extract(a).shape

        return feature_shape


class Predictor():

    def __init__(self, zarr_root, zarr_group, fold_list, model_list, conv2d=True, transpose=False, **kwargs):
        'Initialization'

        # Zarr dataset handling
        zarr_root = zarr.open_group(str(zarr_root), mode='r')

        if type(model_list) is not list:
            model_list = [model_list]

        if type(fold_list) is not list:
            fold_list = [fold_list]

        self.model_list = model_list
        self.num_models = len(self.model_list)

        self.norm_data = [None] * self.num_models
        for i, model in enumerate(model_list):

            # Get cross-validation fold metadata
            zarr_fold = zarr_root[f'{zarr_group}/folds/fold{fold_list[i]}']

            # Normalization data
            self.norm_data[i] = {}
            self.norm_data[i]['mean'] = zarr_fold['norm_data']['mean'][:]
            self.norm_data[i]['std'] = zarr_fold['norm_data']['std'][:]
        
        # Get metadata
        self.metadata = zarr_root[zarr_group].attrs.asdict()
        self.scene_labels = self.metadata['scene_labels']

        # Set features dimensions
        self.set_dim(transpose, conv2d)

    def predict(self, features):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        y_preds = [None] * self.num_models
        for i in range(self.num_models):
            y_preds[i] = self._single_predict(features, i)

        y_pred = np.mean(y_preds, axis=0).argmax(axis=-1)

        # return y_pred

        scene_pred = np.array(self.scene_labels)[y_pred]

        return scene_pred

    def _single_predict(self, features, index=0):
        
        X = self.normalize(features, index)

        model = keras.models.load_model(self.model_list[index])
        
        predictions = model.predict(X, verbose=1)

        return predictions

    def normalize(self, features, index=0):

        num_samples = features.shape[0]

        # Fetch features and normalize them
        Xnorm = (features - self.norm_data[index]['mean']) / self.norm_data[index]['std']

        # Transpose if needed
        if self.transpose:
            Xnorm = np.transpose(Xnorm, axes=[0,2,1])

        # Generate data
        X = np.reshape(Xnorm, [num_samples, *self.dim])

        return X

    def set_dim(self, transpose, conv2d):
        
        # Set features dimensions
        self.dim = self.metadata['feature_shape']

        # Set transposed output
        self.transpose = transpose
        if self.transpose:
            self.dim = (*self.dim[::-1],)

        # Set Conv2D input
        self.conv2d = conv2d
        if self.conv2d:
            self.dim = (*self.dim, 1)

