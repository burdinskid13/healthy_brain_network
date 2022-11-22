import warnings
warnings.filterwarnings("ignore")


def run():
    from hbn.features import build_features
    from hbn.constants import Defaults

    # make parent spec file for features
    parent_spec = build_features.make_parent_spec(out_dir=Defaults.FEATURE_DIR)
    
    # make feature and target spec files
    build_features.make_feature_specs(parent_spec, out_dir=Defaults.FEATURE_DIR)
    build_features.make_target_specs(parent_spec, out_dir=Defaults.FEATURE_DIR)

if __name__ == "__main__":
    run()


    