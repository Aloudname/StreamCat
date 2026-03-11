# -*- coding: utf-8 -*-
# config.py using munch to load .yaml to an easy-accessible object.
import yaml
from munch import Munch

def load_config(yaml_file='config/config.yaml'):
    """Load streaming pipeline config (separate from training config)."""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return Munch.fromDict(yaml.safe_load(f))
