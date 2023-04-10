import click
import warnings
from hbn.constants import Defaults
#warnings.filterwarnings("ignore")

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

@click.command()
@click.option("--results_dir", required=True)
@click.option("--participant_specs", required=True, cls=PythonLiteralOption, default=[])

def run(
    results_dir,
    participant_specs=['participant_spec2.json', 'participant_spec3.json', 'participant_spec4.json', 'participant_spec5.json', 'participant_spec6.json']):

    """evaluate models

    Args:
        results_dir (str): fullpath to top-level results dir. for example '../out-localspec-<>'
        participant_specs (list of str): list of participant specs filenames ['participant_spec2.json'] etc.
    """
    from pathlib import Path
    import glob
    import os
    from hbn import io
    from hbn.models import predictive_modeling
    from hbn.constants import Defaults

    print('evaluating phenotypic models')

    # loop over participant specs
    df_all = pd.DataFrame()
    for spec in participant_specs:
        df = predictive_modeling.evaluation(results_dir=results_dir, test_spec=spec)  
        df_all = pd.concat([df_all, df], axis=1)

    fpath = os.path.join(Path(results_dir).parent, f'evaluation_all.csv')
    if os.path.isfile(fpath):
        df = pd.read_csv(fpath)
        df_all = pd.concat([df_all, df], axis=1)
    print(f'writing out evaluation results: {fpath} to disk')
    df_all.to_csv(fpath, index=False)


if __name__ == "__main__":
    run()
