import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from pathlib import Path
from hbn.specs import base_specs
from hbn import io

def _save_dict_to_json(spec_info, base_info, out_dir):

    # save out dict to spec
    io.make_dirs(out_dir)
    for k, v in spec_info.items():

        # update dictionary with base info
        if len(base_info)>0:
            v.update(base_info)

        # write out participant specs
        fpath = os.path.join(out_dir, f'{k}-spec.json')
        io.save_json(fpath, v)
    
    return v


def run(out_dir):
    """
    Runs the main function of the program.

    This function imports the necessary modules and packages, and then performs a series of actions to make 
    target specs, feature specs, participant specs, and pydraml base specs. The function does not take any parameters and does 
    not return any values.

    Args:
        out_dir (str): directory where specs will be saved

    Returns:
    None
    """

    # make base specs
    base_info, spec_info = base_specs.pydraml()
    pydraml_info = _save_dict_to_json(spec_info, base_info, out_dir)
    print(f'created pydraml specs, saved to {out_dir}', flush=True)

    # make participant specs
    base_info, spec_info = base_specs.participants()
    participants_info = _save_dict_to_json(spec_info, base_info, out_dir)
    print(f'created participant specs, saved to {out_dir}', flush=True)

    # make feature specs
    base_info, spec_info = base_specs.features()
    feature_info = _save_dict_to_json(spec_info, base_info, out_dir)
    print(f'created feature specs, saved to {out_dir}', flush=True)

    # make target specs
    base_info, spec_info = base_specs.targets()
    target_info = _save_dict_to_json(spec_info, base_info, out_dir)
    print(f'created target specs, saved to {out_dir}', flush=True)


if __name__ == "__main__":
    run()


    