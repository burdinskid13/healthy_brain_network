import warnings
warnings.filterwarnings("ignore")
import click
import ast

from hbn.constants import Defaults

def make_model_spec(
            filename,
            target_spec,
            feature_spec,
            participants,
            out_dir=Defaults.MODEL_SPEC_DIR
             ):
    """make model specs (json spec files) from the feature specs stored in `FEATURE_DIR`.
    model specs are saved out to `out_dir`
    Args:
        filename (str): full path to features file. Saved in MODEL_SPEC_DIR
        target_spec (str): full path to target spec file. SAVED IN FEATURE_DIR
        feature_spec (str): full path to feature spec file. SAVED IN FEATURE_DIR
        participants (list of str): list of fullpaths to participant files. Example ['../train_participants-ADHD.csv', '../train_participants-No_Diagnosis_Given.csv']
        out_dir (str): full path to model spec output directory. default is `Defaults.MODEL_SPEC_DIR`
    Returns:
        full outpath to `model_spec` JSON
    """
    import re
    import os
    from hbn import io
    from pathlib import Path

    target_info = io.read_json(target_spec)
    feature_info = io.read_json(feature_spec)

    # get participant filenames
    participant_fnames = []
    for participant in participants:
        participant_fnames.append(Path(participant).name)

    # define spec file
    spec_info = {
                "filename": Path(filename).name,
                "feature_spec": feature_info,
                "target_spec": target_info,
                "participants": participant_fnames,
                "x_indices":[],
                "target_vars": [target_info['outname']],
                "group_var": None,
                "n_splits": 50,
                "test_size": 0.2,
                "clf_info": [
                            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}],
                            ],
                "permute": [True, False],
                "gen_feature_importance": True,
                "gen_permutation_importance": False,
                "permutation_importance_n_repeats": 5,
                "permutation_importance_scoring": "accuracy",
                "gen_shap": False,
                "nsamples": "auto",
                "l1_reg": "aic",
                "plot_top_n_shap": 10,
                "metrics": ['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
            }
    
    # write out model spec to disk ../model_specs/
    spec_name = 'classifier-' + '_'.join(re.split(r'_|,|/| ', feature_info['measures'])) + '-' + target_info['outname'] + '-spec.json'
    io.save_dict_as_JSON(fpath=os.path.join(out_dir, spec_name), data_dict=spec_info)
    print(f'save model specs to file for {spec_name}')

    return os.path.join(out_dir, spec_name)


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.option("--target")
@click.option('--participants', cls=PythonLiteralOption, default=[])

def run(
    target='target_DX_01_Cat_binarize-spec.json',
    participants=['train_participants-Specific_Learning_Disorder_with_Impairment_in_Reading.csv', 'train_participants-No_Diagnosis_Given.csv']
    ):
    import glob
    import os
    from hbn.constants import Defaults
    from hbn.models import predictive_modeling

    # get all features
    features = glob.glob(os.path.join(Defaults.FEATURE_DIR, '*features*'))

    # get full paths to participants
    all_participants = []
    for participant in participants:
        all_participants.append(os.path.join(Defaults.MODEL_SPEC_DIR, 'train', participant))

    for feature in features:
        predictive_modeling.make_spec(
            feature_spec=os.path.join(Defaults.FEATURE_DIR, feature),
            target_spec=os.path.join(Defaults.FEATURE_DIR, target),
            participants=all_participants,
            out_dir=Defaults.MODEL_SPEC_DIR
            )
    

if __name__ == "__main__":
    run()


    