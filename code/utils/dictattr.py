#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

from types import SimpleNamespace

def dictattr(data):
    if not isinstance(data, dict):
        raise ValueError("data must be dict object.")

    def _dict2attr(d):
        _d = {}
        for key, item in d.items():
            if isinstance(item, dict):
                _d[key] = _dict2attr(item)
            else:
                _d[key] = item
        return SimpleNamespace(**_d)

    return _dict2attr(data)
