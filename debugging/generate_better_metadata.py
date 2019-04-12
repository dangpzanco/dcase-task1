import pandas as pd
import pathlib
import platform

# Dataset root path
if 'deep-learning-1-vm' in platform.node(): # Server Linux (Google)
    db_path = pathlib.Path('/home/dangpzanco/datasets/TUT-urban-acoustic-scenes-2018-development/')
elif 'Linux' in platform.system(): # Linux (Avell)
    db_path = pathlib.Path('/home/daniel/ufsc/tcc/datasets/TUT-urban-acoustic-scenes-2018-development/')
elif 'Daniel' in platform.node(): # Windows (ASUS)
    db_path = pathlib.Path('D:/Documents/UFSC/TCC/datasets/TUT-urban-acoustic-scenes-2018-development/')
elif int(platform.release()) == 7: # PC do LINSE
    db_path = pathlib.Path('E:/zanco/TCC/datasets/TUT-urban-acoustic-scenes-2018-development/')

db_path = pathlib.Path('/media/linse/dados/zanco/TCC/datasets/TUT-urban-acoustic-scenes-2018-development/')

def get_metadata(db_path=None):

    # Default path
    if db_path is None:
        db_path = 'TUT-urban-acoustic-scenes-2018-development'
    
    db_path = pathlib.Path(db_path)

    # Download and extract dataset if it doesn't exist
    if not db_path.exists():
        import dcase_util as du

        db = du.datasets.TUTUrbanAcousticScenes_2018_DevelopmentSet(data_path=db_path, included_content_types=['all'])
        db.initialize()
        db.show()

    meta_path = pathlib.Path('dataset_metadata.csv')

    if not meta_path.exists():
        # Open CSV metadata
        metadata = pd.read_csv(str(db_path / 'meta.csv'), sep='\t')
        train_meta = pd.read_csv(str(db_path / 'evaluation_setup/fold1_train.txt'), sep='\t', names=['filename', 'scene_label'])

        # Find all the training examples in the complete metadata
        example_type = metadata['filename'].isin(train_meta['filename'])

        # Split the dataset files between training and testing
        example_type[example_type == True] = 'train'
        example_type[example_type == False] = 'test'

        # Rearrange columns
        cols = metadata.columns.tolist()
        cols.insert(1,'example_type')

        # Insert the column for example types (train and test)
        metadata['example_type'] = example_type.values
        metadata = metadata[cols]

        # Remove useless column (for Task 1A)
        metadata = metadata.drop('source_label', axis=1)

        # Keep filename simpler
        metadata['filename'] = metadata['filename'].str.replace('audio/','').str.replace('.wav','')

        # Save metadata to CSV
        metadata.to_csv('dataset_metadata.csv', index=False, sep='\t')
    else:
        metadata = pd.read_csv(meta_path, sep='\t')

    print(metadata.to_string())

    return metadata
