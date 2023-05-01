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
@click.option("--features", required=False, cls=PythonLiteralOption, default=[])
@click.option("--participant_spec", required=False)
@click.option("--target", required=False)
@click.option("--pydraml_spec", required=False)
@click.option("--out_dir", required=False)


def run(
    features=['features-all-all-all-all-spec.json'],
    participant_spec='spec-adhd-01.json',
    target='target_DX_01_Cat_new_binarize-spec.json',
    pydraml_spec='pydraml_spec2.json',
    out_dir=Defaults.MODEL_SPEC_DIR,
    ):
    """Make phenotype model(s) using the following:`feature specs`, `targets`, `participant_spec`, `pydraml_spec`  
    
    Args:
        features (list of str): optional input arg of full path to feature specs.
        participant_spec (list of str): fullpath to participant spec. 
        target (str): fullpath to target spec.
        pydraml_spec (str): fullpath to pydraml spec. 
        out_dir (str): directory where model spec files will be saved.
    """
    from pathlib import Path
    import glob
    import os
    from hbn.models import predictive_modeling

    print('making phenotype models', flush=True)

    if not isinstance(features, list):
        features = [features]

    if out_dir is None:
        out_dir = Defaults.MODEL_SPEC_DIR

    for feature in features:
        if Path(feature).name!='features-parent_spec.json':
                predictive_modeling.make_model(
                                    feature_spec=feature,
                                    target_spec=target,
                                    pydraml_spec=pydraml_spec,
                                    participant_spec=participant_spec,
                                    drop_identifiers=True,
                                    out_dir=out_dir
                                    )
    print('finished making model specs', flush=True)

if __name__ == "__main__":
    run()


    