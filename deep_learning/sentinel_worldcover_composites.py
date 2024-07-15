"""Sentinel-1 and Sentinel-2 annual composites for 2020 and 2021."""

from typing import Any, Callable, Optional, Union

from rasterio.crs import CRS

from torchgeo.datasets import RasterDataset

class SentinelComposites(RasterDataset):
    is_image = True
    # filename_regex = r"""
    #     ^T(?P<tile>\d{{2}}[A-Z]{{3}})
    #     _(?P<date>\d{{4}}_v)
    #     _(?P<band>B[018][\dA]) # How to adapt this?
    #     (?:_(?P<resolution>{}m))?
    #     \..*$
    # """
    # filename_regex = r"""
    #     _(?P<date>\d{{4}}_v)
    # """
    # date_format = "%Y"

class Sentinel2RGBNIR(SentinelComposites):
    """Sentinel-2 RGBNIR annual composites for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*S2RGBNIR*"
    

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

class Sentinel2SWIR(SentinelComposites):
    """Sentinel-2 SWIR annual composites for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*SWIR*"
    
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

class Sentinel2NDVI(SentinelComposites):
    """Sentinel-2 NDVI annual composites for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*NDVI*"
    
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

class Sentinel1(SentinelComposites):
    """Sentinel-1 VV, VH, VV/VH annual composites for 2020 and 2021.

    """

    filename_glob = "ESA_WorldCover_*S1VVVHratio*"
    
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
