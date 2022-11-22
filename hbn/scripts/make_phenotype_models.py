import warnings
warnings.filterwarnings("ignore")

from hbn.constants import Defaults

def run(
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
        # there have to be more than one column, more than one unique target, more than 100 participants
        conditionals = all((dataframe.shape[1]>1, len(dataframe[target_name].unique())>1, dataframe.shape[0]>100))
        
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

if __name__ == "__main__":
    run()


    