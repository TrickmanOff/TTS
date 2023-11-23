"""
an abstraction over local storage of configs and checkpoints
only these classes know about exact paths where the files are stored
"""
import torch

import os
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Union

from lib.utils.util import write_json


logger = logging.getLogger(__name__)


class RunStorage:
    CONFIG_FILENAME = 'config.json'
    CHECKPOINT_EXT = '.pth'

    def __init__(self, exp_name: str, run_name: str, run_dirpath: Union[str, Path]):
        self.exp_name = exp_name
        self.run_name = run_name
        self._run_directory = Path(run_dirpath)
        self._is_removed = False

    def save_config(self, config) -> None:
        assert not self._is_removed
        config_filepath = self.get_config_filepath()
        write_json(config, config_filepath)
        logger.info(f'Saving config for run {self.get_run_id()}')

    def save_checkpoint(self, checkpoint_name: str, state: Dict[str, Any]) -> None:
        assert not self._is_removed
        checkpoints_dirpath = self.get_checkpoints_dirpath()
        checkpoint_filepath = checkpoints_dirpath / (checkpoint_name + self.CHECKPOINT_EXT)
        torch.save(state, checkpoint_filepath)
        logger.info(f'Saving checkpoint {checkpoint_name} for run {self.get_run_id()}')

    def get_config_filepath(self) -> Path:
        assert not self._is_removed
        return self._run_directory / self.CONFIG_FILENAME

    def get_checkpoints_dirpath(self) -> Path:
        assert not self._is_removed
        return self._run_directory

    def get_checkpoints_filepaths(self) -> Dict[str, Path]:
        """
        returns {checkpoint_name: checkpoint_filepath}
        """
        assert not self._is_removed
        checkpoints: Dict[str, Path] = {}
        checkpoints_dirpath = self.get_checkpoints_dirpath()
        for filename in os.listdir(checkpoints_dirpath):
            if filename.endswith(self.CHECKPOINT_EXT):
                checkpoint_name = os.path.splitext(filename)[0]
                checkpoints[checkpoint_name] = checkpoints_dirpath / filename
        return checkpoints

    def get_run_id(self) -> str:
        assert not self._is_removed
        return f'{self.exp_name}:{self.run_name}'

    def remove_run(self) -> None:
        """
        After calling this method the run cannot be used again
        """
        shutil.rmtree(self._run_directory)


class ExperimentsStorage:
    def __init__(self, experiments_dir: Union[str, Path]):
        self._experiments_dir = Path(experiments_dir)

    def get_run(self, exp_name: str, run_name: str, create_run_if_no: bool = False) -> RunStorage:
        run_dirpath = self._experiments_dir / exp_name / run_name
        if not os.path.exists(run_dirpath):
            if create_run_if_no:
                os.makedirs(run_dirpath, exist_ok=True)
            else:
                raise RuntimeError(f'{run_dirpath} does not exist')
        if not os.path.isdir(run_dirpath):
            raise RuntimeError(f'Not a directory {run_dirpath}')
        return RunStorage(exp_name, run_name, run_dirpath)

    def get_all_runs(self) -> Dict[str, Dict[str, RunStorage]]:
        runs = defaultdict(dict)
        for exp_dirname in os.listdir(self._experiments_dir):
            exp_dirpath = self._experiments_dir / exp_dirname
            if os.path.isdir(exp_dirpath):
                for run_dirname in os.listdir(exp_dirpath):
                    run_dirpath = exp_dirpath / run_dirname
                    if os.path.isdir(run_dirpath):
                        runs[exp_dirname][run_dirname] = RunStorage(exp_dirname, run_dirname, run_dirpath)
        return dict(runs)
