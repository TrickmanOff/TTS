import random
from random import shuffle
from typing import Sequence, Optional

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from lib.postprocessing.base_postprocessor import BasePostprocessor
from lib.spectrogram_decoder.base_decoder import BaseDecoder
from lib.trainer.base_trainer import BaseTrainer
from lib.metric.base_metric import BaseMetric
from lib.logger.utils import plot_spectrogram_to_buf
from lib.utils import inf_loop, MetricTracker, get_lr


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics: Sequence[BaseMetric],
            optimizer,
            config,
            device,
            dataloaders,
            spectrogram_decoder: Optional[BaseDecoder] = None,
            postprocessor: Optional[BasePostprocessor] = None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device, lr_scheduler)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.spectrogram_decoder = spectrogram_decoder
        self.postprocessor = postprocessor
        self.lr_scheduler = lr_scheduler
        self.log_step = config["trainer"].get("log_step", 50)

        self.train_metrics = MetricTracker(
            "loss", "grad norm",
            *criterion.get_loss_parts_names(),
            *[m.name for m in self.metrics],
            writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss",
            *criterion.get_loss_parts_names(),
            *[m.name for m in self.metrics],
            writer=self.writer
        )
        self.accumulated_grad_steps = 0
        self.accumulate_grad_steps = config["trainer"].get("accumulate_grad_steps", 1)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["true_mel_spec", "true_duration", "true_pitch", "true_energy", "phonemes_tokens"]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch) -> dict:
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.criterion.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )

                self.writer.add_scalar(
                    "learning rate",
                    self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else get_lr(self.optimizer)
                )

                self._log_predictions(**batch)
                # self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(**batch)
        if is_train and self.accumulated_grad_steps == 0:
            self.optimizer.zero_grad()
        batch.update(outputs)

        # criterion returns a dict, in which the final loss has a key 'loss'
        losses = self.criterion(**batch)
        batch.update(losses)
        if self.postprocessor is not None:
            batch = self.postprocessor(**batch)
        if is_train:
            (batch["loss"] / self.accumulate_grad_steps).backward()
            self.accumulated_grad_steps += 1
            if self.accumulated_grad_steps % self.accumulate_grad_steps == 0:
                self._clip_grad_norm()
                self.optimizer.step()
                self.accumulated_grad_steps = 0
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        for loss_part in losses:
            metrics.update(loss_part, batch[loss_part].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.criterion.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            # self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            pred_log_duration,
            pred_mel_spec,
            id=None,
            true_duration=None,
            true_mel_spec=None,
            mel_length=None,
            examples_to_log=3,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

        batch_size = len(pred_log_duration)
        # pred_waves = self.spectrogram_decoder.decode_as_wave(pred_mel_spec)  # (B, T)
        # true_waves = self.spectrogram_decoder.decode_as_wave(true_mel_spec) if true_mel_spec is not None else [None] * batch_size
        if true_mel_spec is None:
            true_mel_spec = [None] * batch_size
        if true_duration is None:
            true_duration = [None] * batch_size
        if mel_length is None:
            mel_length = [None] * batch_size
        if id is None:
            id = range(batch_size)

        tuples = list(zip(id, text, pred_mel_spec, pred_log_duration, true_mel_spec, true_duration, mel_length))
        shuffle(tuples)
        rows = {}

        self._log_stats(pred_mel_spec=pred_mel_spec, true_mel_spec=true_mel_spec, **kwargs)

        for id, text, pred_mel, pred_log_duration, true_mel, true_duration, mel_length \
                in tuples[:examples_to_log]:
            rows[id] = {
                "text": text,
            }
            if self.spectrogram_decoder is not None:
                if true_duration is not None:
                    pred_mel = pred_mel[:, :mel_length]
                    true_mel = true_mel[:, :mel_length]
                    mels = torch.stack((pred_mel, true_mel), dim=0)
                    # true_wave = self.spectrogram_decoder.decode_as_wave(true_mel.unsqueeze(0)).squeeze(0)
                    # pred_wave = self.spectrogram_decoder.decode_as_wave(pred_mel.unsqueeze(0)).squeeze(0)
                    pred_wave, true_wave = self.spectrogram_decoder.decode_as_wave(mels)
                    rows[id]['ref wave'] = self._create_audio_for_writer(true_wave)
                    rows[id]['pred wave (using given alignment)'] = self._create_audio_for_writer(pred_wave)
                else:
                    pred_mel_frames_duration = torch.exp(pred_log_duration).round().long().sum()
                    pred_mel = pred_mel[:, :pred_mel_frames_duration]
                    pred_wave = self.spectrogram_decoder.decode_as_wave(pred_mel.unsqueeze(0)).squeeze(0)
                    rows[id]['pred wave (w/o given alignment)'] = self._create_audio_for_writer(pred_wave)

        table = pd.DataFrame.from_dict(rows, orient="index")\
                            .reset_index().rename(columns={'index': 'mix_id'})
        self.writer.add_table("predictions", table)

    def _create_audio_for_writer(self, audio: torch.Tensor, length=None):
        audio = audio.detach().cpu().squeeze()
        if length is not None:
            audio = audio[:length]
        return self.writer.create_audio(audio, sample_rate=self.config["common"]["sr"])

    def _log_stats(self, **batch):
        idx = random.randint(0, batch['true_pitch'].shape[0] - 1)
        self.writer.add_multiline_plot('pitch', {'true pitch': batch['true_pitch'][idx],
                                                 'pred pitch': batch['pred_pitch'][idx]})
        self.writer.add_multiline_plot('energy', {'true energy': batch['true_energy'][idx],
                                                  'pred energy': batch['pred_energy'][idx]})
        pred_mel_spec_img = PIL.Image.open(plot_spectrogram_to_buf(batch['pred_mel_spec'][idx]))
        true_mel_spec_img = PIL.Image.open(plot_spectrogram_to_buf(batch['true_mel_spec'][idx]))
        self.writer.add_image("pred spectrogram", ToTensor()(pred_mel_spec_img))
        self.writer.add_image("true spectrogram", ToTensor()(true_mel_spec_img))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
