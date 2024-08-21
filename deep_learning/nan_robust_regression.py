from torchgeo.trainers import PixelwiseRegressionTask
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import WeightsEnum

from torchgeo.datasets import unbind_samples


class NanRobustPixelWiseRegressionTask(PixelwiseRegressionTask):

    def __init__(
        self,
        model: str = "resnet50",
        backbone: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        num_outputs: int = 1,
        num_filters: int = 3,
        loss: str = "mse",
        nan_value: float = -32767.0,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        """Initialize a new RegressionTask instance.

        Args:
            model: Name of the
                `timm <https://huggingface.co/docs/timm/reference/models>`__ or
                `smp <https://smp.readthedocs.io/en/latest/models.html>`__ model to use.
            backbone: Name of the
                `timm <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or
                `smp <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone
                to use. Only applicable to PixelwiseRegressionTask.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            num_outputs: Number of prediction outputs.
            num_filters: Number of filters. Only applicable when model='fcn'.
            loss: One of 'mse' or 'mae'.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to linear probe
                the regression head. Does not support FCN models.
            freeze_decoder: Freeze the decoder network to linear probe
                the regression head. Does not support FCN models.
                Only applicable to PixelwiseRegressionTask.

        .. versionchanged:: 0.4
           Change regression model support from torchvision.models to timm

        .. versionadded:: 0.5
           The *freeze_backbone* and *freeze_decoder* parameters.

        .. versionchanged:: 0.5
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.
        """
        self.nan_value = nan_value
        super().__init__(model=model,backbone=backbone,weights=weights,in_channels=in_channels,num_outputs=num_outputs,num_filters=num_filters,loss=loss,lr=lr,patience=patience,freeze_backbone=freeze_backbone,freeze_decoder=freeze_decoder)

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        if loss == "mse":
            self.criterion: nn.Module = nn.MSELoss(reduction=None)
        elif loss == "mae":
            self.criterion = nn.L1Loss(reduction=None)
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'mse' or 'mae' loss."
            )

    def _remove_nan_in_target(y_hat: Tensor, y: Tensor, nan_value: float) -> (Tensor, Tensor):
        indices = torch.nonzero(y==nan_value)
        y_hat[indices] = nan_value
        y = y[y!=nan_value]
        y_hat = y_hat[y_hat!=nan_value]
        return y_hat, y



    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        y_hat, y = _remove_nan_in_target(y_hat,y,self.nan_value)
        loss: Tensor = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics)

        return loss


    def validation_step(
            self, batch: Any, batch_idx: int, dataloader_idx: int = 0
        ) -> None:
            """Compute the validation loss and additional metrics.

            Args:
                batch: The output of your DataLoader.
                batch_idx: Integer displaying index of this batch.
                dataloader_idx: Index of the current dataloader.
            """
            x = batch["image"]
            # TODO: remove .to(...) once we have a real pixelwise regression dataset
            y = batch[self.target_key].to(torch.float)
            y_hat = self(x)
            if y_hat.ndim != y.ndim:
                y = y.unsqueeze(dim=1)
            y_hat, y = _remove_nan_in_target(y_hat,y,self.nan_value)
            loss = self.criterion(y_hat, y)
            self.log("val_loss", loss)
            self.val_metrics(y_hat, y)
            self.log_dict(self.val_metrics)

            if (
                batch_idx < 10
                and hasattr(self.trainer, "datamodule")
                and hasattr(self.trainer.datamodule, "plot")
                and self.logger
                and hasattr(self.logger, "experiment")
                and hasattr(self.logger.experiment, "add_figure")
            ):
                try:
                    datamodule = self.trainer.datamodule
                    if self.target_key == "mask":
                        y = y.squeeze(dim=1)
                        y_hat = y_hat.squeeze(dim=1)
                    batch["prediction"] = y_hat
                    for key in ["image", self.target_key, "prediction"]:
                        batch[key] = batch[key].cpu()
                    sample = unbind_samples(batch)[0]
                    fig = datamodule.plot(sample)
                    if fig:
                        summary_writer = self.logger.experiment
                        summary_writer.add_figure(
                            f"image/{batch_idx}", fig, global_step=self.global_step
                        )
                        plt.close()
                except ValueError:
                    pass


    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        y_hat, y = _remove_nan_in_target(y_hat,y,self.nan_value)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)        