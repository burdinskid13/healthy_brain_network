from hbn.features import build_features
from hbn import io
import pandas as pd
import click
import os

from hbn.constants import Defaults
from hbn.scripts import script_features
from hbn.visualization import visualize

import warnings
warnings.filterwarnings("ignore")

@click.command()
@click.option("--spec_file")

def run(spec_file='features-Child_Measures-Language_Tasks-all-DX_01_Cat_factorize-spec.json'):
    """this function makes a <features>.csv file using information given by `spec_file`
    dataframe is saved to ../features/<filename> given by `spec_file`
    """
    # load spec file
    spec_fpath = os.path.join(Defaults.BASE_DIR, "features", spec_file)
    spec_info = io.read_json(spec_fpath)
    
    # get features if they haven't already been created
    feature_path = os.path.join(Defaults.BASE_DIR, "features", spec_file.replace('-spec.json', '.csv'))
    if not os.path.isfile(feature_path):
        script_features.run(spec_file=spec_file)
    
    # load features
    df = pd.read_csv(os.path.join(feature_path))
    
    visualize.umap_embeddings(dataframe=df[df.columns[1:]], target=df[spec_info['target']])

    

    