import warnings
warnings.filterwarnings("ignore")
from hbn.scripts import firstlevel_model, secondlevel_model, test_model
from hbn.constants import Defaults

import os

def run(
    firstlevel_models,
    target_spec='target-Diagnosis-spec.json',
    pydraml_spec='pydraml3-spec.json',
    participants=['Reading-all'],
    model_parent='reading_jan_2024'
    ):

    ## loop over first level model
    for model in firstlevel_models: 

        ## loop over participants
        for participant in participants:

            # define model dir
            model_dir = os.path.join(Defaults.MODEL_DIR, f'{model_parent}/participants-{participant}', model)

            # run firstlevel model 
            firstlevel_model.run(
                                 participant_spec=f'participants-{participant}-spec.json', 
                                 feature_spec=f'features-{model}-spec.json',
                                 target_spec=target_spec, pydraml_spec=pydraml_spec, 
                                 spec_dir=Defaults.MODEL_SPECS_DIR, 
                                 data_dir=os.path.join(Defaults.INTERIM_DIR, 'phenotypes'), 
                                 model_dir=model_dir)
            
            # run secondlevel model
            firstlevel_model_dir = os.path.join(model_dir, model)
            secondlevel_model.run(model_dir=firstlevel_model_dir)

    # evaluate trained models on test data
    secondlevel_models = [f'{model}_secondlevel' for model in firstlevel_models]
    firstlevel_models.extend(secondlevel_models)

    for model in firstlevel_models:

        for participant in participants: 

            # define model dir
            model_dirn = os.path.join(Defaults.MODEL_DIR, f'{model_parent}/participants-{participant}', model)

            test_model.run(model_dir=model_dirn, 
                           model_spec=os.path.join(model_dirn, 'model_spec-test.json')
                           )

if __name__== "__main__":
    run()