"""Spanish vegetation Normalized Surface Digital Model"""

from typing import Any, Callable, Optional, Union

from rasterio.crs import CRS

from torchgeo.datasets import RasterDataset
from torchgeo.datasets.utils import BoundingBox
from typing import Any, cast


class PNOAnDSMV(RasterDataset):
    """Spanish vegetation Normalized Surface Digital Model

    """

    is_image = False 
    filename_glob = "PNOA_*"
    # dtype = float32

    filename_regex = r'PNOA_(?P<date>\d{4})'
    date_format = "%Y"

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

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        This implementation overloads the default one by making it compatible with Kornia AgumentationSequential because 
        TorchGeo AugmentationSequential is deprecated since version 0.4. Specifically the sample dict passed to transform
        should not contain keys that are not within kornia.augmentation.DataKey enum, like 'crs'. To comply with the
        requirement transform is only applied to the 'image' or 'mask' key, and the sample  dict is completed afterwards.
        Note that using Korni AugmentationSequential with a RasterDataset requires setting data_keys to None.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        if self.separate_files:
            data_list: list[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if 'band' in match.groupdict():
                            start = match.start('band')
                            end = match.end('band')
                            filename = filename[:start] + band + filename[end:]
                    filepath = os.path.join(directory, filename)
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            data = torch.cat(data_list)
        else:
            data = self._merge_files(filepaths, query, self.band_indexes)

        sample = {}

        data = data.to(self.dtype)
        if self.is_image:
            sample['image'] = data
        else:
            sample['mask'] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

       # sample['crs'] = self.crs
       # sample['bbox'] = query    

        return sample    
