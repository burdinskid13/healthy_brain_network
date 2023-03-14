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
@click.option("--features", required=True, cls=PythonLiteralOption, default=[])
@click.option("--target", required=False)
@click.option("--pydraml_spec", required=False)
@click.option("--out_dir", required=False)
@click.option('--participant_spec', required=False)

def run(
    features=[],
    target='target_DX_01_Cat_new_binarize-spec.json',
    pydraml_spec='pydraml_spec2.json',
    out_dir=Defaults.MODEL_SPEC_DIR,
    participant_spec='participant_spec1.json'
    ):
    """Make phenotype model(s) using the following:`feature specs`, `targets`, `participant_spec`, `pydraml_spec`  
    
    Args:
        features (list of str): optional input arg of feature spec filenames (listed in `Defaults.FEATURE_DIR`). If empty list, all features are used to create model specs. Default is empty list.
        target (str): target spec filename. should be stored in 'Defaults.FEATURE_DIR'
        pydraml_spec (str): pydraml spec filename. should be stored in `out_dir`
        out_dir (str): where model spec files will be saved. Default is Defaults.MODEL_SPEC_DIR
        participant_spec (str): participant spec filename. should be stored in `out_dir`
    """
    from pathlib import Path
    import glob
    import os
    from hbn.models import predictive_modeling

    # get all features
    if not features:
        features = glob.glob(os.path.join(Defaults.FEATURE_DIR, '*features*'))

    if not isinstance(features, list):
        features = [os.path.join(Defaults.FEATURE_DIR, feature) for feature in features]

    if out_dir is None:
        out_dir = Defaults.MODEL_SPEC_DIR

    for feature in features:
        if Path(feature).name!='features-parent_spec.json':
            predictive_modeling.make_model(
                                feature_spec=os.path.join(Defaults.FEATURE_DIR, feature),
                                target_spec=os.path.join(Defaults.FEATURE_DIR, target),
                                pydraml_spec=os.path.join(Defaults.MODEL_SPEC_DIR, pydraml_spec),
                                participant_spec=os.path.join(Defaults.MODEL_SPEC_DIR, participant_spec),
                                drop_identifiers=True,
                                out_dir=out_dir
                                )

if __name__ == "__main__":
    run()


    