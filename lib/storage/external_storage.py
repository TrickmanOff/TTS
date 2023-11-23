import logging
from abc import abstractmethod

from lib.storage.experiments_storage import RunStorage


logger = logging.getLogger(__name__)


def try_to_call_function(function, times: int = 3):
    exc = None
    for _ in range(times):
        try:
            function()
            return
        except Exception as e:
            exc = e
    logging.warning(f"Failed to use external storage: {exc}")


class ExternalStorage:
    def import_config(self, run_storage: RunStorage) -> None:
        try_to_call_function(lambda: self._import_config(run_storage))

    def import_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        try_to_call_function(lambda: self._import_checkpoint(run_storage, checkpoint_name))

    def export_config(self, run_storage: RunStorage) -> None:
        try_to_call_function(lambda: self._export_config(run_storage))

    def export_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        try_to_call_function(lambda: self._export_checkpoint(run_storage, checkpoint_name))

    @abstractmethod
    def _import_config(self, run_storage: RunStorage) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _import_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _export_config(self, run_storage: RunStorage) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _export_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def list_content(self) -> str:
        raise NotImplementedError()
