
feature_spec = '/om2/user/maedbh/healthy_brain_network/model_specs/features-Child-externalizing-spec.json'
target_spec = '/om2/user/maedbh/healthy_brain_network/model_specs/target-Diagnosis-spec.json'
participant_spec ='/om2/user/maedbh/healthy_brain_network/model_specs/participants-Reading-fluent-readers-female-spec.json'
pydraml_spec='/om2/user/maedbh/healthy_brain_network/model_specs/pydraml3-spec.json'
data_dir='/om2/user/maedbh/hbn_data/interim/phenotypes/'
out_dir='/om2/user/maedbh/hbn_data/interim/models/test'

from hbn.scripts import make_firstlevel_model
#from hbn.scripts.make_firstlevel_model import *

make_firstlevel_model.run(feature_spec, target_spec, participant_spec, pydraml_spec, data_dir, out_dir)

