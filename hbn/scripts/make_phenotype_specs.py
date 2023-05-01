import warnings
warnings.filterwarnings("ignore")


def run():
    from hbn.features import build_features
    from hbn.models import pydra_ml_specs
    from hbn.data import participant_specs
    from hbn.constants import Defaults

    print('making phenotypic specs...', flush=True)

    # make parent spec file for features
    parent_spec = build_features.make_parent_spec(out_dir=Defaults.FEATURE_DIR)
    
    # make feature and target spec files
    build_features.make_feature_specs(parent_spec, out_dir=Defaults.FEATURE_DIR)
    build_features.make_target_specs(parent_spec, out_dir=Defaults.FEATURE_DIR)

    # make participant specs
    participant_specs.make_specs()

    # make pydraml base specs
    pydra_ml_specs.make_specs(out_dir=Defaults.MODEL_SPEC_DIR)

if __name__ == "__main__":
    run()


    