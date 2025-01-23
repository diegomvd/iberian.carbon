from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import BoundingBox
from typing import Any
import torch

class PNOAVegetation(RasterDataset):
    """
    Spanish vegetation Normalized Surface Digital Model
    """
    is_image = False 
    filename_glob = "PNOA_*"

    filename_regex = r'PNOA_(?P<date>\d{4})'
    date_format = "%Y"

    nan_value = -32767.0

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        result = super().__getitem__(query)
        return result

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the dataset (overrides the dtype of the data file via a cast).

        Defaults to float32 if :attr:`~RasterDataset.is_image` is True, else long.
        Can be overridden for tasks like pixel-wise regression where the mask should be
        float32 instead of long.

        Returns:
            the dtype of the dataset

        .. versionadded:: 0.5
        """
        return torch.float32



 
