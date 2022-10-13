from hbn import io
from hbn.scripts import run_phenotype_workflow

def run():  
    spec_file = '/Users/maedbhking/Documents/healthy_brain_network/features/features-Child_Measures-all_domains-all_measures-spec.json'
    spec_info = io.read_json(spec_file)

    df = run_phenotype_workflow.make_features_from_spec(spec_info)

    return df