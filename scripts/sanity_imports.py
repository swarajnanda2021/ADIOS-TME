"""Import-only sanity check, runnable on Mac.

Confirms the new files are syntactically valid and that the imports we wrote
match each other. Modules that depend on cluster-assembled `cellvit/*` files
will fail on Mac; that's expected and handled.
"""

import importlib
import os
import sys

# Make `adios_cellvit/`, `models/`, etc. importable when this script is run
# as `python scripts/sanity_imports.py` from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Should import cleanly on Mac (no cellvit/* dependency).
mac_modules = [
    'adios_cellvit.channel_selector',
]

# Depend on cellvit/* (will fail on Mac; OK on cluster).
cluster_modules = [
    'adios_cellvit.adios_backbone',       # ADIOS-TME internals only, but they
                                          # may pull in CUDA/xformers — handled
                                          # in the loop below.
    'adios_cellvit.adios_cellvit_model',  # depends on cellvit.models
]


def main() -> int:
    for mod in mac_modules:
        try:
            importlib.import_module(mod)
            print(f'OK   {mod}')
        except Exception as e:
            print(f'FAIL {mod}: {e}')
            return 1

    # Deps that the ADIOS-TME repo (or cellvit assembly) needs but a Mac dev
    # install usually lacks. Missing any of these = expected SKIP, not FAIL.
    cluster_only_deps = ('cellvit', 'xformers', 'timm')

    for mod in cluster_modules:
        try:
            importlib.import_module(mod)
            print(f'OK   {mod}')
        except ImportError as e:
            msg = str(e)
            if any(dep in msg for dep in cluster_only_deps):
                print(f'SKIP {mod} ({msg.splitlines()[0]} — expected on Mac)')
            else:
                print(f'FAIL {mod}: {e}')
                return 1
        except Exception as e:
            print(f'FAIL {mod}: {e}')
            return 1

    print('\nMac-side sanity: passed (within expected limitations).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
