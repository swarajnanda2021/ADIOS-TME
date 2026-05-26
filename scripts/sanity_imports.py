"""Import-only sanity check, runnable on Mac.

Confirms the new files are syntactically valid and that the imports we wrote
match each other. Modules that depend on cluster-assembled `cellvit/*` files
will fail on Mac; that's expected and handled. Heavy CV / cluster-only deps
that a stock Mac install usually lacks (cv2, albumentations, timm, xformers,
cellvit) are likewise treated as expected SKIPs.
"""

import importlib
import os
import sys

# Make `adios_cellvit/`, `models/`, etc. importable when this script is run
# as `python scripts/sanity_imports.py` from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Should import cleanly on Mac (assuming the optional CV deps below are
# installed). If they aren't, we SKIP rather than FAIL — see comments below.
mac_modules = [
    'adios_cellvit.channel_selector',
    'adios_cellvit.pannuke_dataset',
]

# Depend on cellvit/* or other cluster-only modules. Expected to SKIP on Mac.
cluster_modules = [
    'adios_cellvit.adios_backbone',       # ADIOS-TME internals (pulls in timm + xformers)
    'adios_cellvit.adios_cellvit_model',  # depends on cellvit.models
]

# Deps a Mac dev install often lacks (heavy CV stack + the cluster-assembled
# cellvit package). Missing any of these = expected SKIP, not FAIL.
OPTIONAL_DEPS = ('cellvit', 'xformers', 'timm', 'cv2', 'albumentations')


def _classify(mod: str) -> str:
    """Return 'ok' / 'skip:<reason>' / 'fail:<reason>'."""
    try:
        importlib.import_module(mod)
        return 'ok'
    except ImportError as e:
        msg = str(e)
        if any(dep in msg for dep in OPTIONAL_DEPS):
            return f'skip:{msg.splitlines()[0]}'
        return f'fail:{e}'
    except Exception as e:
        return f'fail:{e}'


def main() -> int:
    for mod in mac_modules + cluster_modules:
        outcome = _classify(mod)
        if outcome == 'ok':
            print(f'OK   {mod}')
        elif outcome.startswith('skip:'):
            print(f'SKIP {mod} ({outcome[5:]} — expected on Mac)')
        else:
            print(f'FAIL {mod}: {outcome[5:]}')
            return 1

    print('\nMac-side sanity: passed (within expected limitations).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
