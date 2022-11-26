import warnings
warnings.filterwarnings("ignore")
import click
import ast

from hbn.constants import Defaults


def make_spec(
    feature_spec,
    target_spec,
    participants,
    out_dir=Defaults.MODEL_SPEC_DIR
    ):
    """make model spec file using the following: 'feature_spec', 'target_spec' and 'participants'

    Grabs all existing feature specs and targets and `participants` and create a model spec file

    Args: 
        feature_spec (str): fullpath to feature spec
        target_spec (str): fullpath to target spec
        participants (list of str): For example: ['../train_participants-ADHD.csv', '../train_participants-No_Diagnosis_Given.csv']
        out_dir (str): directory where model spec and feature file should be saved
    Returns:
        model_spec (str): full path to model spec
    """
    import os
    from hbn import io
    import random
    from hbn.models import first_level_modeling as first_level
    from hbn.constants import Defaults

    random_number = round(random.random()*1000000000)
    filename = f'model_features_{random_number}.csv'
    print(f'trying to make new filename: {filename}')

    model_spec = None
    try:
        # make multiple model specs using features, target, and participant specs 
        dataframe = first_level.make_model_features(
                            feature_spec=feature_spec, 
                            target_spec=target_spec,
                            participants=participants
                            )
        # load target info
        target_name = io.read_json(target_spec)['outname']

        # set certain conditionals for model spec to be run and model features to be created
        # there have to be more than one column, more than one unique target, more than 100 participants and fewer than 1000 features
        conditionals = all((dataframe.shape[1]>1, len(dataframe[target_name].unique())>1, dataframe.shape[0]>100, dataframe.shape[1]<1000))
        
        if conditionals:  
            # make model spec file
            model_spec = first_level.make_model_spec(filename,
                                        target_spec=target_spec,
                                        feature_spec=feature_spec,
                                        participants=participants,
                                        out_dir=Defaults.MODEL_SPEC_DIR
                                        )
            dataframe.to_csv(os.path.join(out_dir, f'model_features_{random_number}.csv'), index=False)

        else:
            print(f'model spec not created for {filename} because one of the following conditions was not met: more than 1 feature, more than one unique target, more than 100 participants')
    except:
        print(f'failed to make model specs for {filename}')

    return model_spec


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

    # get all features
    features = glob.glob(os.path.join(Defaults.FEATURE_DIR, '*features*'))

    # get full paths to participants
    all_participants = []
    for participant in participants:
        all_participants.append(os.path.join(Defaults.MODEL_SPEC_DIR, 'train', participant))

    for feature in features:
        make_spec(
            feature_spec=os.path.join(Defaults.FEATURE_DIR, feature),
            target_spec=os.path.join(Defaults.FEATURE_DIR, target),
            participants=all_participants,
            out_dir=Defaults.MODEL_SPEC_DIR
            )
    

if __name__ == "__main__":
    run()


    