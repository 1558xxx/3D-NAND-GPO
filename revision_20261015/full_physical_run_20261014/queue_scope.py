"""Authorised retained-review scope; excluded work never counts as completed."""
from full_common import HERE, read_json

PROFILE='retained_reviews_20261014'
REQUIRED_KINDS={'classical','main','regularization','cost','analysis','delivery'}
EXCLUDED_KINDS={'ablation','extra_gpu','extra_cpu'}

def retained_scope():
    return read_json(HERE/'scope_profile.json',{}).get('profile')==PROFILE

def required_job(job):
    return not retained_scope() or job[1] in REQUIRED_KINDS

def scope_description():
    return read_json(HERE/'scope_profile.json',{})
