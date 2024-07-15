"""Spanish vegetation Normalized Surface Digital Model"""

from typing import Any, Callable, Optional, Union

from rasterio.crs import CRS

from torchgeo.datasets import RasterDataset


class PNOAnDSMV(RasterDataset):
    """Spanish vegetation Normalized Surface Digital Model

    """

    is_image = False # False
    filename_glob = "NDSM-*-COB*"
    dtype = float32

    def __init__(
        self,
        paths: Union[str, list[str]] = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load, here
                the collection of individual zip files for each tile should be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``paths``
            RuntimeError: if dataset is missing

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        self.paths = paths

        self._verify()

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)


    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing
        """
        # Check if the extracted files already exists
        if self.files:
            return

        raise RuntimeError(
            f"Dataset not found in `paths={self.paths!r}` "
            "either specify a different `paths` or make sure you "
            "have manually downloaded dataset tiles as suggested in the documentation."
        )