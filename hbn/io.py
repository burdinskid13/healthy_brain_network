def load_json(fpath):
    import json

    f = open(fpath) 
  
    # returns JSON object as a dict 
    return json.load(f) 


def save_json(fpath, dict):
    import json

    with open(fpath, 'w') as fp:
       #json.dump(dict, fp,  indent=4)
       json.dump(dict, fp)


def make_dirs(fpath):
    import os

    if not os.path.exists(fpath):
        print(f"creating {fpath}")
        os.makedirs(fpath)