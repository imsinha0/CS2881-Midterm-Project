import json
import re
import os
import random
import numpy as np
from datetime import datetime
from typing import Tuple, List

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        js = json.load(f)
    return js


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    return txt

def read_raw_data_dir(raw_data_dir, recursive=True) -> List[str]:
    """only read txt files"""
    data = []
    if recursive:
        for root, dirs, files in os.walk(raw_data_dir):
            for f in files:
                if "txt" not in f:
                    continue
                full_path = os.path.join(root, f)
                d = read_txt(full_path)
                data.append(d)
    else:
        raise NotImplementedError
    
    return data
