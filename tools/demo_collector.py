"""The collector, as started by tools/demo_video.sh. Not for real sites.

It is the same collector code. The only difference is where it looks for
vendor register maps: a temporary copy in which the demo has shortened the
Felicity map's poll interval from 30 seconds to 5, so readings move on
camera. On a real box the interval stays as the map says (the adapter does
not let a site config override it), which is why this is done here and not
in the adapter.
"""

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edge"))

import collector.adapters.vendor_map as vendor_map  # noqa: E402

vendor_map.VENDOR_MAPS_DIR = Path(os.environ["DEMO_VENDOR_MAPS_DIR"])
runpy.run_module("collector.main", run_name="__main__")
