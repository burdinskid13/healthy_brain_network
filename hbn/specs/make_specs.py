import os
import re
from hbn import io
from hbn.constants import Defaults
from hbn.specs import base_specs

def make_pydraml_specs(
    out_dir=Defaults.MODEL_SPEC_DIR,
    n_splits=5, 
    test_size=0.2
    ):

    # get pydraml base
    clf_info, base_info = base_specs.pydralml_base(n_splits=n_splits, test_size=test_size)

    # loop over classifies and save out pydra-ml specs
    for name,clf in clf_info.items():

        clf = {'clf_info': clf}
        clf.update(base_info)

        # write out pydra-ml specs
        fpath = os.path.join(out_dir, f'pydraml_{name}.json')
        io.save_dict_as_JSON(fpath, clf)


def make_participant_specs(out_dir=os.path.join(Defaults.MODEL_SPEC_DIR, 'participant_specs')):

    # get spec info
    spec_info = base_specs.participant_base(out_dir=out_dir)

    # loop over classifies and save out specs
    io.make_dirs(out_dir)
    for name, spec in spec_info.items():

        # write out participant specs
        fpath = os.path.join(out_dir, f'{name}.json')
        io.save_dict_as_JSON(fpath, spec)


def make_parent_specs(out_dir=Defaults.FEATURE_DIR):
    """ create parent spec files and save to disk
    """

    # get feature bases
    features = base_specs.features()

    outpaths = []
    for k,v in features.items():

        # get spec info
        base_spec = base_specs.parent_features_base(out_dir=out_dir)

        # get target base and add to base spec
        targets = base_specs.targets()
        base_spec.update({'target': targets})

        # add each feature set to base spec
        base_spec.update({'features': v})

        outpath = os.path.join(out_dir, k, 'features-parent_spec.json')
        io.make_dirs(os.path.join(out_dir, k))
        outpaths.append(outpath)
        io.save_dict_as_JSON(outpath, base_spec)
    
    return outpaths


def make_target_specs(parent_spec, out_dir=Defaults.FEATURE_DIR):
    """make target sets (json spec files)

    Args: 
        parent_spec (str or dict): fullpath to `features-parent-spec.json` or data dictionary loaded from file
        out_dir (str): save to path. default is `Defaults.FEATURE_DIR`
    Returns:
        saves feature spec files (.json) to `FEATURE_DIR` and returns list of feature specs
    """

    if isinstance(parent_spec, str):
        parent_spec = io.read_json(parent_spec)
    
    targets = parent_spec["target"]

    spec_files = []
    for data in targets:
        
        spec_filename = 'target_' + data["outname"]

        # define target spec file
        spec_info = {
                    "assessment": data["assessment"],
                    "domain": data["domain"],
                    "measure": data["measure"],
                    "target_column": data["target_column"],
                    "features_to_ignore": data["features_to_ignore"],
                    "transform": data["transform"], 
                    "outname": data["outname"],
                    "clf_info": parent_spec['preprocessing']['clf_info'], 
                    }

        # save json to `FEATURE_DIR`
        spec_fpath = os.path.join(out_dir, spec_filename + '-spec.json')
        io.save_dict_as_JSON(fpath=spec_fpath, data_dict=spec_info)
        print(f'spec file and features saved to disk for {spec_filename}')


def make_feature_specs(
        parent_spec,
        out_dir=Defaults.FEATURE_DIR,
        features_to_ignore=['features-Clinical_Measures-domains-Clinical_Diagnosis-Diagnosis_ClinicianConsensus', 'features-Clinical_Measures-all-all-all']
        ):
    """make feature sets (json spec files)

    Args: 
        parent_spec (dict): parent spec info (output from `make_parent_spec`)
        out_dir (str): save to path. default is `Defaults.FEATURE_DIR`
        featuers_to_ignore (list of str): list of features to ignore
    Returns:
        saves feature spec files (.json) to `FEATURE_DIR` and returns list of feature specs
    """
    from hbn.features import build_features

    if isinstance(parent_spec, str):
        parent_spec = io.read_json(parent_spec)

    # loop over feature info
    for spec in parent_spec['features']:

        # clean up domain folder name (remove superfluous spaces - should match directory)
        if spec['domains'] is not None:
            domains_parsed = re.split(r'_|,|/| ', spec['domains'])
            while("" in domains_parsed):
                domains_parsed.remove("") 
            spec['domains'] = '_'.join(domains_parsed)

        # get datadic
        datadic = build_features.get_datadic(abbrev=spec['abbrevs'])
        
        # figure out whether preprocessing will be done (set in parent_spec)
        preprocessing = None
        if parent_spec['preprocessing']['preprocess']:
            preprocessing = parent_spec['preprocessing']

        # define feature spec file
        spec_info = {
                    "assessment": spec['assessment'],
                    "domains": spec['domains'],
                    "measures": spec['measures'],
                    "abbrevs": spec['abbrevs'],
                    "datadic": datadic,
                    "add_features": spec['add_features'],
                    "filter_features": spec['filter_features'],
                    "preprocessing": preprocessing
                    }
        
        # make unique filename for each feature spec
        spec_filename = _make_filename(spec)
        
        if spec_filename not in features_to_ignore:

            # save json to `FEATURE_DIR`
            spec_fpath = os.path.join(out_dir, spec_filename + '-spec.json')
            io.save_dict_as_JSON(fpath=spec_fpath, data_dict=spec_info)
            print(f'spec file saved to disk for {spec_filename}')


def _make_filename(data):
    """make filename for feature spec

    Args: 
        data (dict):
    Returns:
        spec_file (str): spec filename
    """

    # define spec filename
    vals = []
    for val in ['features', 'assessment', 'domains', 'measures', 'abbrevs']:
        if val in data.keys() and data[val] is not None:
            vals.append('_'.join(re.split(r'_|,|/| ', data[val])))
        else:
            vals.append(val)
    spec_file = '-'.join(vals)

    return spec_file