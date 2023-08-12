import warnings
warnings.filterwarnings("ignore")
import click

@click.command()
@click.option("--old_dir", required=True)
@click.option("--new_dir", required=True)

def run(old_dir, new_dir):
    import os
    import glob
    import pandas as pd
    from hbn import io
    import shutil
    from pathlib import Path
    from hbn.features.feature_selection import secondlevel_feature_selection

    # make new directory
    io.make_dirs(new_dir)

    # copy files from old model directory to new directory
    feature_spec = glob.glob(os.path.join(old_dir, '**/*feature_importance*'), recursive=True)[0]
    feature_csv = glob.glob(os.path.join(old_dir, '**/model_features_*'), recursive=True)[0]
    model_spec = glob.glob(os.path.join(old_dir, '**/*.json'), recursive=True)[0]

    # copy over files to new directory
    for file in [feature_spec, feature_csv, model_spec]:
        shutil.copy(file, new_dir)

    # make new model spec based on second level feature selection
    model_specs_new, spec_names = secondlevel_feature_selection(new_dir)

    # save to file in new directory
    for (model_spec_new, spec_name) in zip(model_specs_new, spec_names):
        io.save_dict_as_JSON(fpath=os.path.join(new_dir, spec_name), data_dict=model_spec_new)

    # remove classifier feature importance file from new directory
    # remove old model spec file from new directory
    os.remove(os.path.join(new_dir, Path(feature_spec).name))
    os.remove(os.path.join(new_dir, Path(model_spec).name))
    
if __name__ == "__main__":
    run()