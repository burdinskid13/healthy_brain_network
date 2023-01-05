import warnings
warnings.filterwarnings("ignore")
import click
import ast
from hbn.constants import Defaults

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.option('--features', required=False)
@click.option("--target", required=False)
@click.option("--pydraml_spec", required=False)
@click.option("--out_dir", required=False)
@click.option('--participants', cls=PythonLiteralOption, default=[])

def run(
    features=None,
    target='target_DX_01_Cat_binarize-spec.json',
    pydraml_spec='pydraml_spec2.json',
    out_dir=Defaults.MODEL_SPEC_DIR,
    participants=['train_participants-Specific_Learning_Disorder_with_Impairment_in_Reading.csv', 'train_participants-No_Diagnosis_Given.csv']
    ):
    """Make phenotype model(s) using the following:`feature specs`, `targets`, `participants`, `pydraml_spec`  
    
    Args:
        features (list of str or None): optional input arg. Default is None. If None, all features are used to create model specs
        target (str): target spec filename. should be stored in 'Defaults.FEATURE_DIR'
        pydraml_spec (str): pydraml spec filename. should be stored in `spec_dir`
        out_dir (str): where model spec files will be saved. Default is Defaults.MODEL_SPEC_DIR
        participants (list of str): participant filenames
    """
    import glob
    import os
    from hbn.models import predictive_modeling

    # get all features
    if features is None:
        features = glob.glob(os.path.join(Defaults.FEATURE_DIR, '*features*'))

    if not isinstance(features, list):
        features = [features]

    if out_dir is None:
        out_dir = Defaults.MODEL_SPEC_DIR

    # get full paths to participants
    all_participants = []
    for participant in participants:
        all_participants.append(os.path.join(Defaults.MODEL_SPEC_DIR, 'train', participant))

    for feature in features:
        model_spec, model_features = predictive_modeling.make_model(
                                    feature_spec=os.path.join(Defaults.FEATURE_DIR, feature),
                                    target_spec=os.path.join(Defaults.FEATURE_DIR, target),
                                    pydraml_spec=os.path.join(Defaults.MODEL_SPEC_DIR, pydraml_spec),
                                    participants=all_participants,
                                    out_dir=out_dir
                                    )
    return model_spec, model_features
    

if __name__ == "__main__":
    run()


    