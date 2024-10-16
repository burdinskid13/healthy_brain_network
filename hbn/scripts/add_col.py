import pandas as pd
from hbn import io
from hbn.constants import Defaults
import os
import glob


def add_col(fname, data_dir):
    # load model spec info

    info = io.load_json(os.path.join(data_dir, 'model_spec-train.json'))

    # load dataframe
    df = pd.read_csv(fname, engine='python')
    df['feature_filename'] = info['feature_info']['filename'].replace('.csv', '')

    return df

def run(model='reading_augustB'):
    
    # get all models run in `model`
    models = glob.glob(os.path.join(Defaults.MODEL_DIR, model, '*', '*'))

    for m in models:

        summary_fpath = os.path.join(m, 'model-summary.csv')
        feature_fpath = os.path.join(m, 'feature_importance.csv')

        # read model summary and features into pd dataframe
        if os.path.isfile(summary_fpath):
            df_summary = add_col(fname=summary_fpath, data_dir=m)
            df_summary.to_csv(summary_fpath, index=False)
        
        if os.path.isfile(feature_fpath):
            df_feature = add_col(fname=feature_fpath, data_dir=m)
            df_feature.to_csv(feature_fpath, index=False)


if __name__ == "__main__":
    run()

