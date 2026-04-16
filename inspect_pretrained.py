#!/usr/bin/env python
"""Inspect alignn.pretrained module"""
import os
os.chdir("/root/autodl-tmp")
import inspect
import alignn.pretrained as p

# Print full source of key functions
for fn in ['get_figshare_model', 'get_prediction', 'get_multiple_predictions']:
    print(f"\n{'='*60}")
    print(f"FUNCTION: {fn}")
    print('='*60)
    src = inspect.getsource(getattr(p, fn))
    print(src[:3000])
