import click

@click.command()
@click.option("--cachedir")

def run(
    cachedir='/home/maedbh/.cache/pydra-ml/cache-wf/'
    ):
    """
    FIRST STEP: parse raw data (if not already parsed), create summary files for clinical diagnoses  
    SECOND STEP: make feature and target spec files using the source spec file `features/features-parent_spec.json`  
    THIRD STEP: make model spec files using feature, target, and participant files 
    FOURTH STEP: run modeling routine: uses `https://github.com/nipype/pydra-ml` 
    
    Args: 
        cachedir (str): full path to model cache directory.
            on openmind I use: '/home/maedbh/.cache/pydra-ml/cache-wf/
            on local I use '/Users/maedbhking/pydra-ml/cache-wf/'
            on savio I use: '/global/scratch/users/maedbhking/bin/pydra-ml/cache-wf/'
    """
    import os
    from hbn.scripts import preprocess_phenotype
    from hbn.scripts import make_phenotype_specs
    from hbn.scripts import make_phenotype_models
    from hbn.constants import Defaults
    from hbn.models import predictive_modeling

    # FIRST STEP
    preprocess_phenotype.run()

    # SECOND STEP
    make_phenotype_specs.run()

    # THIRD STEP
    feature_spec = 'features-Parent_Measures-Demographic_Questionnaire_Measures-Extended_Strengths_and_Weaknesses_Assessment_of_Normal_Behavior-Parent_Report-spec.json'
    target_spec =  'target_DX_01_Cat_binarize-spec.json'
    pydraml_spec = 'pydraml_spec2.json'

    MODEL_SPEC_TRAIN = os.path.join(Defaults.MODEL_SPEC_DIR, 'train')
    model_spec, _ = make_phenotype_models.run(
                                        feature_spec=os.path.join(Defaults.FEATURE_DIR, feature_spec),
                                        target_spec=os.path.join(Defaults.FEATURE_DIR, target_spec),
                                        pydraml_spec=os.path.join(Defaults.MODEL_SPEC_DIR, pydraml_spec),
                                        participants = [MODEL_SPEC_TRAIN + '/train_participants-ADHD.csv', 
                                                        MODEL_SPEC_TRAIN + '/train_participants-No_Diagnosis_Given.csv']
                                        )

    # FOURTH STEP
    predictive_modeling.run_pydra_ml(
        model_spec=os.path.join(Defaults.MODEL_SPEC_DIR, model_spec), 
        spec_dir= Defaults.MODEL_SPEC_DIR, 
        out_dir=os.path.join(Defaults.TEST_DIR, 'test_data'),
        cachedir=cachedir
        )


if __name__ == "__main__":
    run()



