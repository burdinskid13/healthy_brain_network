import warnings
warnings.filterwarnings("ignore")


def get_phenotype_inputs(assessment=['Child Measures', 'Parent Measures', 'Teacher Measures']):
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
        w2.options = get_domains()
    w1.observe(update_domain_options, 'value')
    def update_measure_options(change):
        w3.options = build_features.get_measures(assessment=w1.value, domain=change.new)[change.new]
    w2.observe(update_measure_options, 'value')
    return widgets.VBox([w1, w2, w3]) 


def get_participant_inputs(participants=['train', 'test', 'all']):
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
        w2.options =  get_disorder_categories()
    w1.observe(update_domain_options, 'value')
    def update_measure_options(change):
        w3.options = get_disorder(column='DX_01', category=change.new)
    w2.observe(update_measure_options, 'value')
    return widgets.VBox([w1, w2, w3]) 

