#!/usr/bin/env python3

### Helpfull commands ###
# oarsub -l gpu=1,walltime=3:00:00 -I
# module load conda
# conda activate ibs
# cd /home/bfollmer/code/IBS/IBS

# export nnUNet_raw="/srv/storage/epione@storage2.sophia.grid5000.fr/bfollmer/data/AL/nnunet/nnUNet_raw"
# export nnUNet_preprocessed="/srv/storage/epione@storage2.sophia.grid5000.fr/bfollmer/data/AL/nnunet/nnUNet_preprocessed"
# export nnUNet_results="/srv/storage/epione@storage2.sophia.grid5000.fr/bfollmer/data/AL/nnunet/nnUNet_results"
# export nnUNet_compile=False
# export nnUNet_n_proc_DA=1
# export nnUNet_def_n_proc=1

# python src/IBS.py --config configs/ibs.local.json --dataset AMOS --dataset_name_or_id 500 --func init --dim 3 --configuration 3d_fullres --label_manual true --segauto true --nnUNetTrainer nnUNetTrainer_ORGAN_500
# python src/IBS.py --config configs/ibs.local.json --dataset AMOS --dataset_name_or_id 500 --func al --dim 3 --configuration 3d_fullres --label_manual true --segauto true --nnUNetTrainer nnUNetTrainer_ORGAN_500


"""Command-line entry point for IBS experiments."""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
VENDORED_NNUNET_ROOT = SOURCE_ROOT / "tools" / "nnUNet" / "nnUNet"


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def load_config(filepath):
    if filepath is None:
        return {}, REPO_ROOT
    path = Path(filepath).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except FileNotFoundError:
        raise SystemExit(f"Configuration file does not exist: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON configuration {path}: {exc}")
    if not isinstance(config, dict):
        raise SystemExit(f"Configuration must contain a JSON object: {path}")
    return config, path.parent


def resolve_path(cli_value, env_name, config, config_key, config_dir, dataset, default=None):
    """Resolve a path using CLI > environment > config > default precedence."""
    value = cli_value
    relative_to = Path.cwd()
    if value is None:
        value = os.environ.get(env_name)
    if value is None:
        value = config.get(config_key, default)
        relative_to = config_dir
    if value in (None, ""):
        return None
    value = os.path.expandvars(os.path.expanduser(str(value))).format(dataset=dataset)
    path = Path(value)
    if not path.is_absolute():
        path = relative_to / path
    return str(path.resolve())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run Interactive Batch Selection experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="JSON configuration file (see configs/ibs.example.json)")
    parser.add_argument("--dataset", default="CACSPipeR", help="Dataset name")
    parser.add_argument("--dataset_name_or_id", default="150", help="nnU-Net dataset ID")
    parser.add_argument("--strategy", default="INIT", help="Active-learning strategy")
    parser.add_argument("--emulation", default=False, type=str2bool)
    parser.add_argument("--func", default="", choices=("", "init", "al"), help="Function to execute")
    parser.add_argument("--dim", default=3, type=int, choices=(2, 3))
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--budget", default=50, type=int)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--nnUNetTrainer", default="nnUNetTrainer_ORGAN_600")
    parser.add_argument("--targeted", default=True, type=str2bool)
    parser.add_argument("--label_manual", default=True, type=str2bool)
    parser.add_argument("--correction", default=False, type=str2bool)
    parser.add_argument("--segauto", default=False, type=str2bool)
    parser.add_argument("--fastmode", default=False, type=str2bool)
    parser.add_argument("--label_valid", default=False, type=str2bool)
    parser.add_argument("--versionUse", default=None, type=int)
    parser.add_argument("--interactive", default=True, type=str2bool)

    paths = parser.add_argument_group("paths")
    paths.add_argument("--raw-data-dir", "--fp_raw", dest="fp_raw", help="Root containing dataset folders")
    paths.add_argument("--active-dir", dest="fp_active", help="Experiment output directory")
    paths.add_argument("--dataset-data-dir", dest="dataset_data", help="IBS dataset metadata directory")
    paths.add_argument("--modules-dir", dest="fp_modules", help="Directory containing dataset modules")
    paths.add_argument("--nnunet-dir", dest="fp_nnunet", help="Parent directory for nnU-Net data and results")
    paths.add_argument("--manual-dir", dest="fp_manual", help="Manual annotation directory")
    paths.add_argument("--nngeometry-dir", dest="fp_nngeometry", help="Optional nngeometry source directory")
    paths.add_argument("--nnunet-raw", dest="nnUNet_raw", help="nnU-Net raw dataset directory")
    paths.add_argument("--nnunet-preprocessed", dest="nnUNet_preprocessed", help="nnU-Net preprocessed directory")
    paths.add_argument("--nnunet-results", dest="nnUNet_results", help="nnU-Net results directory")
    return parser


def configure_paths(opts, parser):
    config, config_dir = load_config(opts.config)
    dataset = opts.dataset
    mappings = {
        "fp_raw": ("IBS_RAW_DATA_DIR", "raw_data_dir", None),
        "fp_active": ("IBS_ACTIVE_DIR", "active_dir", None),
        "dataset_data": ("IBS_DATASET_DATA_DIR", "dataset_data_dir", None),
        "fp_modules": ("IBS_MODULES_DIR", "modules_dir", str(SOURCE_ROOT / "modules")),
        "fp_nnunet": ("IBS_NNUNET_DIR", "nnunet_dir", None),
        "fp_manual": ("IBS_MANUAL_DIR", "manual_dir", None),
        "fp_nngeometry": ("IBS_NNGEOMETRY_DIR", "nngeometry_dir", None),
        "nnUNet_raw": ("nnUNet_raw", "nnunet_raw", None),
        "nnUNet_preprocessed": ("nnUNet_preprocessed", "nnunet_preprocessed", None),
        "nnUNet_results": ("nnUNet_results", "nnunet_results", None),
    }
    for attribute, (env_name, config_key, default) in mappings.items():
        value = resolve_path(
            getattr(opts, attribute), env_name, config, config_key, config_dir, dataset, default
        )
        setattr(opts, attribute, value)

    if opts.fp_nnunet:
        opts.nnUNet_raw = opts.nnUNet_raw or str(Path(opts.fp_nnunet) / "nnUNet_raw")
        opts.nnUNet_preprocessed = opts.nnUNet_preprocessed or str(Path(opts.fp_nnunet) / "nnUNet_preprocessed")
        opts.nnUNet_results = opts.nnUNet_results or str(Path(opts.fp_nnunet) / "nnUNet_results")
        opts.fp_active = opts.fp_active or str(Path(opts.fp_nnunet) / dataset / "AL")
    if opts.fp_modules:
        opts.dataset_data = opts.dataset_data or str(Path(opts.fp_modules) / dataset / "data")

    if opts.func:
        required = ("fp_raw", "fp_active", "dataset_data", "fp_nnunet", "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
        missing = [name for name in required if not getattr(opts, name)]
        if missing:
            parser.error(
                "missing required paths: "
                + ", ".join(missing)
                + ". Provide --config, CLI path arguments, or environment variables."
            )

    for env_name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        value = getattr(opts, env_name)
        if value:
            os.environ[env_name] = value

    opts.fip_split = str(Path(opts.fp_active) / "INIT" / "INIT_V01" / "model" / "splits_final.json") if opts.fp_active else None
    opts.nnUNetResults = opts.nnUNetTrainer + "__ALUNETPlanner__" + opts.configuration
    opts.ALSamples = [100 for _ in range(10)]


def main():
    parser = build_parser()
    opts = parser.parse_args()
    configure_paths(opts, parser)

    # Make repository-owned packages importable without machine-specific paths.
    for path in (SOURCE_ROOT, VENDORED_NNUNET_ROOT):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)
    if opts.fp_nngeometry and opts.fp_nngeometry not in sys.path:
        sys.path.insert(0, opts.fp_nngeometry)

    # Import after configuring nnU-Net because nnunetv2.paths reads its paths at import time.
    from ALUNET.ALUNET import ALUNET

    strategy_dict = {
        "INIT": 0,
        "FULL": 1,
        "RANDOM": 2,
        "ENTROPY": 3,
        "MCD": 3,
        "RANDOM66": 4,
        "CLASP": 5,
        "IBS": 6,
    }
    if opts.strategy not in strategy_dict:
        parser.error(f"unknown strategy {opts.strategy!r}; choose from {', '.join(strategy_dict)}")

    opts.NumSamplesMaxMCD = 5000
    dataset_samples = {
        "AMOS": {2: [200] + [50] * 50, 3: [100] + [25] * 50},
        "KITS": {2: [200] + [50] * 50, 3: [50] + [25] * 50},
        "ASOCA": {2: [100] + [50] * 50, 3: [25] * 50},
    }
    if opts.dataset in dataset_samples:
        opts.ALSamples = dataset_samples[opts.dataset][opts.dim]

    opts.dataset_name_or_id_init = opts.dataset_name_or_id
    opts.dataset_name_or_id = str(int(opts.dataset_name_or_id) + strategy_dict[opts.strategy])
    if opts.func == "init":
        opts.strategy = "INIT"
        opts.dataset_name_or_id = opts.dataset_name_or_id_init
        alunet = ALUNET(opts)
        alunet.init_dataset(opts)
    else:
        alunet = ALUNET(opts)

    print("opts:", vars(opts))
    if opts.func == "al" and opts.strategy == "FULL":
        alunet.alfull(opts)
    elif opts.func == "al":
        alunet.alround(opts, copy_nnUNet_data=True)


if __name__ == "__main__":
    main()




