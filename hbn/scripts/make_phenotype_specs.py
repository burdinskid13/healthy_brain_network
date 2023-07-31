import warnings
warnings.filterwarnings("ignore")

def run():
    """
    Runs the main function of the program.

    This function imports the necessary modules and packages, and then performs a series of actions to make phenotypic specs,
    target specs, feature specs, participant specs, and pydraml base specs. The function does not take any parameters and does 
    not return any values.

    Parameters:
    None

    Returns:
    None
    """
    import os
    from hbn.constants import Defaults
    from hbn.specs import make_specs
    from pathlib import Path

    print('making phenotypic specs...', flush=True)

    # make parent specs
    parent_specs = make_specs.make_parent_specs(out_dir=Defaults.FEATURE_DIR)

    # loop over parent spec files and make target and feature specs
    for parent_spec in parent_specs:

        make_specs.make_target_specs(parent_spec, out_dir=Path(parent_spec).parent) # make target spec files 
        make_specs.make_feature_specs(parent_spec, out_dir=Path(parent_spec).parent) # make feature spec files

    # make participant spec files
    out_dir = os.path.join(Defaults.MODEL_SPEC_DIR, 'participant_specs')
    make_specs.make_participant_specs(out_dir=out_dir)

    # make pydraml base specs
    make_specs.make_pydraml_specs(out_dir=Defaults.MODEL_SPEC_DIR, n_splits=5, test_size=0.2)

if __name__ == "__main__":
    run()


    