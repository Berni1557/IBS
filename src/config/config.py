"""Portable compatibility configuration for legacy IBS modules.

New code should receive paths from the IBS command-line configuration. These
values remain for legacy modules that import ``CConfig`` directly.
"""

import os
import platform
from pathlib import Path


_source_root = Path(__file__).resolve().parents[1]
_data_root = Path(os.environ.get("IBS_DATA_ROOT", _source_root.parent / "data")).expanduser().resolve()

CConfig = {
    "hostname": platform.node(),
    "device": os.environ.get("IBS_DEVICE", "local"),
    "srcpath": str(Path(os.environ.get("IBS_SRC_ROOT", _source_root)).expanduser().resolve()),
    "datapath": str(_data_root),
    "datapath_source": str(_data_root),
    "replace_dataset": ("", ""),
    "OS": "WIN" if os.name == "nt" else "LINUX",
}
