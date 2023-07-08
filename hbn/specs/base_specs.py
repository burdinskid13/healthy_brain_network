import os
from hbn import io
from hbn.constants import Defaults

def make_pydraml_specs(
    out_dir=Defaults.MODEL_SPEC_DIR,
    n_splits=5, 
    test_size=0.2):

    # get pydraml base
    clf_info, base_info = pydralml_base(n_splits=n_splits, test_size=test_size)

    # loop over classifies and save out pydra-ml specs
    for name,clf in clf_info.items():

        clf.update(base_info)

        # write out pydra-ml specs
        fpath = os.path.join(out_dir, f'pydraml_{name}.json')
        io.save_dict_as_JSON(fpath, clf)


def make_participant_specs(out_dir=os.path.join(Defaults.MODEL_SPEC_DIR, 'participant_specs')):

    # get spec info
    spec_info = participant_base(out_dir=out_dir)

    # loop over classifies and save out specs
    io.make_dirs(out_dir)
    for name, spec in spec_info.items():

        # write out participant specs
        fpath = os.path.join(out_dir, f'{name}.json')
        io.save_dict_as_JSON(fpath, spec)


def make_parent_spec(out_dir=Defaults.FEATURE_DIR):

    # get spec info
    spec_info = parent_features_base(out_dir=out_dir)

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


def pydralml_base(n_splits=5, test_size=0.2):

    base_info = {
        "filename" : None,
        "x_indices" : None,
        "target_vars" : None,
        "clf_info" : clf_info,
        "permute" : [True, False],
        "group_var" : None,
        "n_splits" : n_splits,
        "test_size" : test_size,
        "permute" : [True, False],
        "gen_feature_importance" : True,
        "gen_permutation_importance" : True,
        "permutation_importance_n_repeats" : 5,
        "permutation_importance_scoring" : "accuracy",
        "gen_shap" : False,
        "nsamples" : "auto",
        "l1_reg" : "aic",
        "plot_top_n_shap": 10,
        "metrics" : ['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
        }

    clf_info = {
        'spec1':
        [
            ["sklearn.ensemble", "AdaBoostClassifier"],
            ["sklearn.naive_bayes", "GaussianNB"],
            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}],
            ["sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100}],
            ["sklearn.ensemble", "ExtraTreesClassifier", {"n_estimators": 100, "class_weight": "balanced"}],
            ["sklearn.linear_model", "LogisticRegressionCV", {"solver": "liblinear", "penalty": "l1"}],
            ["sklearn.neural_network", "MLPClassifier", {"alpha": 1, "max_iter": 1000}],
            ["sklearn.svm", "SVC", {"probability": True},
            [{"kernel": ["rbf", "linear"], "C": [1, 10, 100, 1000]}]],
        ],
        'spec2':
        [
         ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]
        ],
        'spec3':
        [
            ["sklearn.feature_selection", "SelectFromModel", {"estimator": "LinearSVC"}],
            ["sklearn.tree", "DecisionTreeClassifier", {"max_depth": 5}]
        ]
        }
    return clf_info, base_info


def participant_base(out_dir=os.path.join(Defaults.MODEL_SPEC_DIR, 'participant_specs')):

    spec_info = {
        'spec-adhd-No_Diagnosis_01':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-asd-No_Diagnosis_01':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-Subtypes_02':
        {'diagnoses': ['ADHD-Combined_Type', 'ADHD-Inattentive_Type'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-depression-No_Diagnosis_01':
        {'diagnoses': ['Depressive_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-anxiety-No_Diagnosis_01':
        {'diagnoses': ['Anxiety_Disorders', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-reading-No_Diagnosis_01':
        {'diagnoses': ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-No_Diagnosis_03':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-No_Diagnosis_04':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-asd-No_Diagnosis_02':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-asd-No_Diagnosis_03':
        {'diagnoses': ['Autism_Spectrum_Disorder', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': 'all',
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-reading-No_Diagnosis_02':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-reading-No_Diagnosis_03':
        {
        "diagnoses": ['Specific_Learning_Disorder_with_Impairment_in_Reading', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-No_Diagnosis_02':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-anxiety-No_Diagnosis_03':
        {
        "diagnoses": ['Anxiety_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-depression-No_Diagnosis_02':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-depression-No_Diagnosis_03':
        {
        "diagnoses": ['Depressive_Disorders', 'No_Diagnosis_Given'],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-age-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [6],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [14],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [16],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [6],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [14],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [16],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-05':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [6],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-07':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [8],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-09':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [10],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-11':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [12],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-13':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [14],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-15':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [16],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-05_06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-07_08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7,8],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-09_10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9,10],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-11_12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11,12],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-13_14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-15_16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'all',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-05_06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-07_08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7,8],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-09_10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9,10],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-11_12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11,12],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-13_14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-15_16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-male-17+':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'male',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-05_06':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [5,6],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-07_08':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [7,8],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-09_10':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [9,10],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-11_12':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [11,12],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-13_14':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [13,14],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-15_16':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [15,16],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-age-female-17':
        {'diagnoses': ['ADHD', 'No_Diagnosis_Given'],
         'split': 'train',
         'age': [17,18,19,20,21,22],
         'sex': 'female',
         'ethnicity': 'all'
        },
        'spec-adhd-anxiety-05':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-depression-06':
        {
        "diagnoses": [
            "ADHD",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-asd-07':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-reading-08':
        {
        "diagnoses": [
            "ADHD",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        'spec-adhd-anxiety-09':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-depression-10':
        {
        "diagnoses": [
            "ADHD",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-asd-11':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-reading-12':
        {
        "diagnoses": [
            "ADHD",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "male",
        "ethnicity": "all"
        },
        'spec-adhd-anxiety-13':
        {
        "diagnoses": [
            "ADHD",
            "Anxiety_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-depression-14':
        {
        "diagnoses": [
            "ADHD",
            "Depressive_Disorders"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-asd-15':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-reading-16':
        {
        "diagnoses": [
            "ADHD",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "female",
        "ethnicity": "all"
        },
        'spec-adhd-multilabel':
        {
        "diagnoses": [
            "ADHD",
            "Autism_Spectrum_Disorder",
            "Anxiety_Disorders",
            "Depressive_Disorders",
            "Specific_Learning_Disorder_with_Impairment_in_Reading"
        ],
        "split": "train",
        "age": "all",
        "sex": "all",
        "ethnicity": "all"
        },
        }
    return spec_info


def parent_features_base(out_dir=Defaults.FEATURE_DIR):

    base_info = {
                "preprocessing": {
                    "preprocess": True,
                    "cols_to_drop": ['EID', 'Comment_ID', 'Unnamed', 'Administration', 'Days_Baseline', 'Data_entry', 'START_DATE', 'Year', 'Site', 'Season', 'Visit_label', 'Study', 'PSCID', 'Release_Number'], # cols to drop while preprocessing
                    "cols_to_ignore": ['Identifiers'], # cols to ignore in the preprocessing routine (column transformation)
                    "upsample": True, # upsample minority class using SMOTE
                    "threshold": False, #threshold dataframe based on some fixed criterion
                    "clf_info": {
                        "numeric": [
                            [
                                "sklearn.impute",
                                "SimpleImputer",
                                {
                                    "strategy": "mean"
                                }
                            ],
                            [
                                "sklearn.preprocessing",
                                "StandardScaler",
                                {}
                            ]
                        ]
                        }
            }
            }
    
    # get target info
    target_info = targets()
    base_info.update(target_info)

    # get feature info
    feature_info = features()
    base_info.update(feature_info)

    return base_info


def targets():
    """hardcode target features
    """
    target_info = {
            "target": [
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01_Cat",
                    "features_to_ignore": ['KSADS', 'Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "binarize",
                    "outname": "DX_01_Cat_binarize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01_Cat_new",
                    "features_to_ignore": ['KSADS', 'Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "binarize",
                    "outname": "DX_01_Cat_new_binarize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01_Cat_new",
                    "features_to_ignore": ['KSADS', 'Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "factorize",
                    "outname": "DX_01_Cat_new_factorize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01_Cat",
                    "features_to_ignore": ['KSADS', 'Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "factorize",
                    "outname": "DX_01_Cat_factorize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01",
                    "features_to_ignore": ['KSADS','Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "binarize",
                    "outname": "DX_01_binarize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "DX_01",
                    "features_to_ignore": ['KSADS','Diagnosis', 'DX_01_Cat', 'DX_01_Cat_new', 'Category', 'DX_01', 'comorbidities'],
                    "transform": "factorize",
                    "outname": "DX_01_factorize"
                    },
                    {"assessment": "Clinical Measures",
                    "domain": None,
                    "measure": "Clinical Diagnosis Demographics",
                    "target_column": "Sex",
                    "features_to_ignore": ['KSADS','Sex'],
                    "transform": "binarize",
                    "outname": "Sex_binarize"
                    }
                ],
            }
    return target_info


def features():
    ### RETURN TO THIS ###
    feature_info = 
            {
            "features": [
                {"assessment": ["Child Measures", "Parent Measures", "Teacher Measures", "Clinical Measures"],
                "domains": "all",
                "measures": "all",
                "abbrevs": 'all',
                "filter": None,
                "add_features": 'Demographic_Features.csv' # add additional features. (str or None)   
                }, 
                {"assessment": ["Child Measures", "Parent Measures", "Teacher Measures", "Clinical Measures"],
                "domains": "all",
                "measures": "all",
                "abbrevs": 'all',
                "filter": "",
                "add_features": 'Demographic_Features.csv' # add additional features. (str or None)   
                }
            ]
            }

    return feature_info
