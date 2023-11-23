"""
Class for importing and exporting models from Google Drive.
"""
import dataclasses
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

from oauth2client.service_account import ServiceAccountCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from lib.storage.experiments_storage import RunStorage
from lib.storage.external_storage import ExternalStorage


logger = logging.getLogger(__name__)

ARCHIVE_FORMAT = 'zip'


def _archive_file(archive_filepath: str, filepath: str) -> str:
    """
    returns archive filepath
    """
    if os.path.isdir(filepath):
        shutil.make_archive(archive_filepath, ARCHIVE_FORMAT, root_dir=filepath)
    else:
        parent_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        shutil.make_archive(archive_filepath, ARCHIVE_FORMAT, root_dir=parent_dir, base_dir=filename)
    return archive_filepath + '.' + ARCHIVE_FORMAT


@dataclasses.dataclass
class RunInfo:
    checkpoints: List[str]
    with_config: bool


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


class GDriveStorage(ExternalStorage):
    CONFIG_FILENAME = 'config.json'
    CHECKPOINT_EXT = '.pth'

    def __init__(self, storage_dir_id: str,
                 client_secrets_filepath: str = './client_secrets.json',
                 key_filepath: Optional[str] = None,
                 gauth=None):
        """
        :param storage_dir_id: the id of the GDrive directory where experiments are stored
        """
        super().__init__()
        if key_filepath is not None:
            assert gauth is None
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                key_filepath,
                scopes = ['https://www.googleapis.com/auth/drive.readonly'],
            )
            gauth = GoogleAuth()
            gauth.credentials = credentials
    
        if gauth is None:
            GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = client_secrets_filepath
            gauth = GoogleAuth()
            url = gauth.GetAuthUrl()
            print(f'Visit the url:\n{url}')
            code = input('Enter the code')
            gauth.Auth(code)
            # gauth.CommandLineAuth()
        self.drive = GoogleDrive(gauth)
        self.storage_dir_id = storage_dir_id

    # def _get_part_name(self, part: ModelParts):
    #     if part not in self.PARTS_NAMES:
    #         raise RuntimeError(f'Model part {part} is not supported by GDriveStorage')
    #     return self.PARTS_NAMES[part]

    def _import_config(self, run_storage: RunStorage) -> None:
        self._download_file(run_storage, self.CONFIG_FILENAME, run_storage.get_config_filepath())
        logger.info(f'Config for run {run_storage.get_run_id()} imported')

    def _import_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        checkpoint_filename = checkpoint_name + self.CHECKPOINT_EXT
        to_filepath = run_storage.get_checkpoints_dirpath() / checkpoint_filename
        self._download_file(run_storage, checkpoint_filename, to_filepath)
        logger.info(f'Checkpoint {checkpoint_name} for run {run_storage.exp_name}:{run_storage.run_name} imported')

    def _export_config(self, run_storage: RunStorage) -> None:
        config_local_filepath = run_storage.get_config_filepath()
        config_local_filename = os.path.basename(config_local_filepath)
        run_drive_dir = self._get_run_dir(run_storage.exp_name, run_storage.run_name)
        self._upload_file(run_drive_dir, config_local_filename, config_local_filepath)
        logger.info(f'Config for run {run_storage.exp_name}:{run_storage.run_name} exported')

    def _export_checkpoint(self, run_storage: RunStorage, checkpoint_name: str) -> None:
        checkpoint_filename = checkpoint_name + self.CHECKPOINT_EXT
        checkpoint_filepath = run_storage.get_checkpoints_dirpath() / checkpoint_filename
        run_drive_dir = self._get_run_dir(run_storage.exp_name, run_storage.run_name)
        self._upload_file(run_drive_dir, checkpoint_filename, checkpoint_filepath)

    def _get_subdir(self, parent_dir_id: str, subdir_name: str) -> str:
        """
        :return: id
        """
        query = f'"{parent_dir_id}" in parents and title="{subdir_name}" and trashed=false'
        files = self.drive.ListFile({'q': query}).GetList()
        if len(files) != 0:
            return files[0]['id']
        else:
            file_metadata = {
                'title': subdir_name,
                'parents': [{'id': parent_dir_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.drive.CreateFile(file_metadata)
            folder.Upload()
            return folder['id']

    def _get_experiment_dir(self, exp_name: str) -> str:
        """
        Creates if it does not exist
        :return: id
        """
        return self._get_subdir(parent_dir_id=self.storage_dir_id, subdir_name=exp_name)

    def _get_run_dir(self, exp_name: str, run_name: str) -> str:
        """
        Creates if it does not exist
        :return: id
        """
        exp_dir = self._get_experiment_dir(exp_name)
        return self._get_subdir(parent_dir_id=exp_dir, subdir_name=run_name)

    # def _import_model_part(self, model_name: str, part: ModelParts, to_filepath: str) -> None:
    #     part_name = self._get_part_name(part)
    #     model_dir = self._get_model_dir(model_name)
    #     with tempfile.TemporaryDirectory() as archive_dir:
    #         downloaded_archive_filepath = os.path.join(archive_dir,
    #                                                    model_name + '.' + ARCHIVE_FORMAT)
    #         query = f'"{model_dir}" in parents and title="{part_name}" and trashed=false'
    #         files = self.drive.ListFile({'q': query}).GetList()
    #         if len(files) == 0:
    #             raise RuntimeError(f'No part {part} found')
    #         archive_file = files[0]
    #         archive_file.GetContentFile(downloaded_archive_filepath)
    #
    #         if os.path.exists(to_filepath) and os.path.isdir(to_filepath):
    #             shutil.unpack_archive(downloaded_archive_filepath, to_filepath, ARCHIVE_FORMAT)
    #         else:
    #             to_dirpath = os.path.dirname(to_filepath)
    #             imported_filename = os.path.basename(to_filepath)
    #             shutil.unpack_archive(downloaded_archive_filepath, to_dirpath, ARCHIVE_FORMAT)
    #             unpacked_filepath = os.path.join(to_dirpath, os.path.basename(downloaded_archive_filepath))
    #             os.rename(unpacked_filepath, os.path.join(to_dirpath, imported_filename))

    def _upload_file(self, to_dir_id: str, to_filename: str, filepath: Union[str, Path]) -> None:
        query = f"'{to_dir_id}' in parents and title='{to_filename}' and trashed=false"
        files = self.drive.ListFile({'q': query}).GetList()
        if len(files) != 0:
            for file in files:
                file.Delete()

        file = self.drive.CreateFile(
            {'title': to_filename, 'parents': [{'id': to_dir_id}]})
        file.SetContentFile(filepath)
        file.Upload()

    def _download_file(self, run_storage: RunStorage, drive_filename: str, to_filepath: Union[str, Path]):
        run_drive_dir = self._get_run_dir(run_storage.exp_name, run_storage.run_name)
        query = f'"{run_drive_dir}" in parents and title="{drive_filename}" and trashed=false'
        files = self.drive.ListFile({'q': query}).GetList()
        if len(files) == 0:
            raise RuntimeError(f'No file {drive_filename} for run {run_storage.get_run_id()}')
        file = files[0]

        if os.path.exists(to_filepath):
            logger.warning(f'File {os.path.basename(to_filepath)} for run {run_storage.get_run_id()} will be overriden')

        file.GetContentFile(to_filepath)
        logger.info(f'Successfully downloaded file {drive_filename} for run {run_storage.get_run_id()}')

    def _export_dir(self, to_dir_id: str, from_dirpath: str) -> None:
        """
        Without archiving, `from_dirpath` must not contain directories
        """
        for filename in os.listdir(from_dirpath):
            filepath = os.path.join(from_dirpath, filename)
            self._upload_file(to_dir_id, filename, filepath)

    def _export_as_archive(self, dir_id: str, archive_name: str, from_path: str) -> None:
        with tempfile.TemporaryDirectory() as archive_dir:
            archive_filepath = _archive_file(os.path.join(archive_dir, archive_name),
                                             filepath=from_path)
            self._upload_file(dir_id, archive_name, archive_filepath)

    # def _export_model_part(self, model_name: str, part: ModelParts, from_path: str) -> None:
    #     part_name = self._get_part_name(part)
    #     model_dir = self._get_model_dir(model_name)
    #
    #     # extra copy for convenience
    #     if part is ModelParts.CONFIG:
    #         config_subdir = self._get_subdir(parent_dir_id=model_dir, subdir_name=self.CONFIG_SUBDIR_NAME)
    #         self._export_dir(config_subdir, from_path)
    #
    #     self._upload_file(model_dir, part_name, from_path)
    #     # self._export_as_archive(model_dir, part_name, from_path)
    #

    def get_available_runs(self) -> Dict[str, Dict[str, RunInfo]]:
        exps_list = self.drive.ListFile(
            {'q': f"'{self.storage_dir_id}' in parents and trashed=false"}).GetList()

        res = {}

        for exp in exps_list:
            runs_list = self.drive.ListFile(
                {'q': f"'{exp['id']}' in parents and trashed=false"}).GetList()
            exp_runs = {}
            for run in runs_list:
                run_info = RunInfo([], False)
                run_files_list = self.drive.ListFile(
                    {'q': f"'{run['id']}' in parents and trashed=false"}).GetList()
                for run_file in run_files_list:
                    run_filename = run_file['title']
                    if run_filename.endswith(self.CHECKPOINT_EXT):
                        checkpoint_name = os.path.splitext(run_filename)[0]
                        run_info.checkpoints.append(checkpoint_name)
                    elif run_filename == self.CONFIG_FILENAME:
                        run_info.with_config = True
                exp_runs[run['title']] = run_info
            res[exp['title']] = exp_runs

        return res

    def list_content(self) -> str:
        runs = self.get_available_runs()
        return json.dumps(runs, indent=4, cls=EnhancedJSONEncoder)

    def _import_model(self, model_name: str, to_dirpath: str) -> None:
        with tempfile.TemporaryDirectory() as archive_dir:
            downloaded_archive_filepath = os.path.join(archive_dir,
                                                       model_name + '.' + ARCHIVE_FORMAT)
            query = f'"{self.storage_dir_id}" in parents and title="{model_name}.{ARCHIVE_FORMAT}" and trashed=false'
            archive_file = self.drive.ListFile({'q': query}).GetList()[0]
            archive_file.GetContentFile(downloaded_archive_filepath)

            shutil.unpack_archive(downloaded_archive_filepath, to_dirpath, ARCHIVE_FORMAT)
