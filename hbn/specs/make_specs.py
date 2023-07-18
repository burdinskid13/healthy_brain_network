import os
from hbn import io
from hbn.constants import Defaults
from hbn.specs import base_specs

def make_pydraml_specs(
    out_dir=Defaults.MODEL_SPEC_DIR,
    n_splits=5, 
    test_size=0.2):

    # get pydraml base
    clf_info, base_info = base_specs.pydralml_base(n_splits=n_splits, test_size=test_size)

    # loop over classifies and save out pydra-ml specs
    for name,clf in clf_info.items():

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


def make_parent_spec(out_dir=Defaults.FEATURE_DIR):

    # get spec info
    spec_info = base_specs.parent_features_base(out_dir=out_dir)

    outpath = os.path.join(out_dir, 'features-parent_spec.json')
    io.save_dict_as_JSON(outpath, spec_info)

    return spec_info


def make_target_specs(parent_spec, out_dir=Defaults.FEATURE_DIR):
    """make target sets (json spec files)

    Args: 
        parent_spec (dict): parent spec info (output from `make_parent_spec`)
        out_dir (str): save to path. default is `Defaults.FEATURE_DIR`
    Returns:
        saves feature spec files (.json) to `FEATURE_DIR` and returns list of feature specs
    """

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
                    "clf_info": spec_info['preprocessing']['clf_info'], 
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

    feature_combinations = _get_feature_combinations(parent_spec)
    
    spec_files = []
    for data in feature_combinations:

        # clean up domain folder name (remove superfluous spaces - should match directory)
        if data['domains'] is not None:
            domains_parsed = re.split(r'_|,|/| ', data['domains'])
            while("" in domains_parsed):
                domains_parsed.remove("") 
            data['domains'] = '_'.join(domains_parsed)
        
        spec_filename = _make_filename(data)

        # get datadic
        datadic = get_datadic(abbrev=data['abbrevs'])

        # add demographics if file is provided by `parent_spec`
        add_features = None
        if parent_spec['add_features'] is not None:
            df_demos = pd.read_csv(os.path.join(Defaults.FEATURE_DIR, parent_spec['add_features']))
            cols_to_include = [col for col in df_demos if 'Identifiers' not in col]
            add_features = {'filename': parent_spec['add_features'],
                            'cols_to_include': cols_to_include
                            }
        
        # figure out whether preprocessing will be done (set in parent_spec)
        preprocessing = None
        if parent_spec['preprocessing']['preprocess']:
            preprocessing = parent_spec['preprocessing']

        # define feature spec file
        spec_info = {
                    "assessment": data['assessment'],
                    "domains": data['domains'],
                    "measures": data['measures'],
                    "abbrevs": data['abbrevs'],
                    "datadic": datadic,
                    "add_features": add_features,
                    "preprocessing": preprocessing
                    }
        
        if spec_filename not in features_to_ignore:

            # save json to `FEATURE_DIR`
            spec_fpath = os.path.join(out_dir, spec_filename + '-spec.json')
            io.save_dict_as_JSON(fpath=spec_fpath, data_dict=spec_info)
            print(f'spec file saved to disk for {spec_filename}')


def _get_feature_combinations(parent_spec):
    """gets combinations of assessment*domain*measure to make feature files from `parent_spec`

    horrible code -- need to rewrite

    Args:
        parent_spec (dict): parent spec info (output from `make_parent_spec`)
    """

    assessments = parent_spec['features']['assessment']
    domains = parent_spec['features']['domains']
    measures = parent_spec['features']['measures']
    abbrevs = parent_spec['features']['abbrevs']

    # check arguments
    if not isinstance(assessments, list):
        assessments = [assessments]
    if (not isinstance(domains, list) and (domains!='all')):
        domains = [domains]
    if (not isinstance(measures, list) and (measures!='all')):
        measures = [measures]

    ## clumsy - should be a cleaner way to write this
    # write out all possible features as models
    spec_info = []
    for assess in assessments:
        if domains=='all':
            domain_names = get_domains(assess)[assess]
            if domain_names is not None:
                domain_names.remove('all')
            elif domain_names is None:
                domain_names = [domain_names]
        for domain in domain_names:
            if measures=='all':
                measure_names = get_measures(assess, domain)[domain]
            for measure in measure_names:
                abbrevs = get_abbrevs(assess, measure)
                for abbrev in abbrevs:
                    spec_info.append({'assessment': assess,
                            'domains': domain,
                            'measures': measure,
                            'abbrevs': abbrev
                            })

        # write out assessment-feature models
        spec_info.append(
            {'assessment': assess,
            'domains': 'all',
            'measures': 'all',
            'abbrevs': 'all'
            })

        # write out all-feature models
        spec_info.append(
            {'assessment': 'all',
             'domains': 'all',
             'measures': 'all',
             'abbrevs': 'all'
             })

    return spec_info


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