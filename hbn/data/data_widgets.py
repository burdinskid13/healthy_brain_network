import warnings
warnings.filterwarnings("ignore")


def get_data(
    phenotype_widget, 
    participant_widget, 
    target_col='DX_01',
    preprocess=True
    ):
    import pandas as pd
    from hbn.features import build_features
    from hbn.data import make_dataset

    out_features = pd.DataFrame()
    out_target = pd.DataFrame()

    # get features
    features = build_features.get_features(
                assessment=phenotype_widget.children[0].value,
                domains=[phenotype_widget.children[1].value],
                measures=list(phenotype_widget.children[2].value),
                min_num_participants=2000,
                incl_data_type=None
                );
    
    if preprocess:
        # preprocess features 
        clf_info={"numeric": [[
                        "sklearn.impute",
                        "SimpleImputer",
                        {"strategy": "mean"}]
    #                                         [
    #                                         "sklearn.preprocessing",
    #                                         "StandardScaler",
    #                                         {}
    #                                     ]
                ]
                }
        features = build_features.preprocess(
                                            dataframe=features,
                                            clf_info=clf_info,
                                            cols_to_ignore=['Identifiers']
                                            )

    # get targets
    targets = build_features.get_targets(target_info = {
                                        "assessment": "Clinical Measures",
                                        "domain": None,
                                        "measure": "Clinical Diagnosis Demographics",
                                        "target_column": target_col,
                                        "transform": None,
                                        "outname": target_col
                                        })

    try:
        # get participant ids
        participants = make_dataset.get_participants(
                                    split=participant_widget.children[0].value, 
                                    disorders=list(participant_widget.children[2].value)
                                    )

        features_target = features.merge(
                                    targets, on='Identifiers').merge(
                                    participants, on='Identifiers')
        # get feature names
        feature_names = [col for col in features_target.columns if target_col not in col]

        out_features = features_target[feature_names]
        out_target = features_target[target_col]
    except: 
        participant_disorder = participant_widget.children[2].value
        print(f'not enough samples in {participant_disorder} to be written to disk, returning empty')

    return out_features, out_target


def get_phenotype_inputs(assessment=['Child Measures', 'Parent Measures', 'Teacher Measures']):
    from hbn.features import build_features
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    
    w1 = widgets.Dropdown(options=assessment)
    w2 = widgets.Dropdown(options=[])
    w3 = widgets.SelectMultiple(
                options=[],
                value=[],
                rows=12,
                description='Measure:',
                disabled=False)
    
    def update_domain_options(change):
        w2.options = build_features.get_domains(change.new)[change.new]
    w1.observe(update_domain_options, 'value')
    def update_measure_options(change):
        w3.options = build_features.get_measures(assessment=w1.value, domain=change.new)[change.new]
    w2.observe(update_measure_options, 'value')
    return widgets.VBox([w1, w2, w3]) 


def get_participant_inputs(participants=['train', 'test', 'all']):
    from hbn.data import make_dataset
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets

    w1 = widgets.Dropdown(options=participants)
    w2 = widgets.Dropdown(options=[])
    w3 = widgets.SelectMultiple(
                options=[],
                value=[],
                rows=15,
                description='Disorder:',
                disabled=False)
    
    def update_domain_options(change):
        w2.options = make_dataset.get_disorder_categories()
    w1.observe(update_domain_options, 'value')
    def update_measure_options(change):
        w3.options =make_dataset.get_disorder(column='DX_01', category=change.new)
    w2.observe(update_measure_options, 'value')
    return widgets.VBox([w1, w2, w3]) 

