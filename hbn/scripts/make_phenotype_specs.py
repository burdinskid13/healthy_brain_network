import warnings
warnings.filterwarnings("ignore")


def run():
    from hbn.constants import Defaults
    from hbn.specs import make_specs

    print('making phenotypic specs...', flush=True)

    # make parent spec
    parent_spec  = make_specs.make_parent_spec(out_dir=Defaults.FEATURE_DIR)

    # make target spec files 
    make_specs.make_target_specs(parent_spec, out_dir=Defaults.FEATURE_DIR)

    # make feature spec files
    make_specs.make_feature_specs(parent_spec, out_dir=Defaults.FEATURE_DIR)

    # make participant spec files
    out_dir = os.path.join(Defaults.MODEL_SPEC_DIR, 'participant_specs')
    make_specs.make_participant_specs(out_dir=out_dir)

    # make pydraml base specs
    make_specs.make_pydraml_specs(out_dir=Defaults.MODEL_SPEC_DIR, n_splits=5, test_size=0.2)

if __name__ == "__main__":
    run()


    