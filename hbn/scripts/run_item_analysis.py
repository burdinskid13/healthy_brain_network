import click
import ast
import warnings
warnings.filterwarnings("ignore")

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

@click.command()
@click.option('--split_age', required=False)
@click.option("--split_sex", required=False)
@click.option('--diagnoses', cls=PythonLiteralOption, default=[], required=False)

def run(diagnoses=['ADHD'],
        split_age=True,
        split_sex=True
        ):
    import glob
    import os
    import itertools
    import datetime
    import pandas as pd
    from hbn.constants import Defaults
    from hbn.models import item_analysis
    
    # get feature specs
    feature_specs = glob.glob(os.path.join(Defaults.FEATURE_DIR, '*Child_Behavior_Checklist*'))
    feature_specs.extend(glob.glob(os.path.join(Defaults.FEATURE_DIR, '*Youth_Self_Report*')))

    sexes = ['male', 'female']
    ages = range(6,11)

    # 
    combos = list(itertools.product(diagnoses, ['all'], ['all']))

    if split_age:
        combos = list(itertools.product(diagnoses, ['all'], ages))
    
    if split_sex:
        combos = list(itertools.product(diagnoses, sexes, ['all']))

    if split_age and split_sex:
        combos = list(itertools.product(diagnoses, sexes, ages))

    df_all = pd.DataFrame()
    for combo in combos:
        diagnosis=combo[0]; sex=combo[1]; age=combo[2]
        df = item_analysis.compare_answers(feature_specs, sex=sex, age=age, diagnoses=[diagnosis], split='all')
        df['diagnosis'] = diagnosis
        df['age'] = age
        df['sex'] = sex
        df_all = pd.concat([df_all, df])
        print(f"adding {diagnosis} for age {age} to dataframe")

    # save dataframe to disk
    fpath = os.path.join(Defaults.INTERIM_DIR, 'questionnaires')
    os.makedirs(fpath)
    ct = datetime.datetime.now()
    ct_name = '_'.join(f'{ct}'.split(' '))
    fname = f'distances_{ct_name}.csv'
    df_all.to_csv(os.path.join(fpath, fname), index=False)

if __name__ == "__main__":
    run()
