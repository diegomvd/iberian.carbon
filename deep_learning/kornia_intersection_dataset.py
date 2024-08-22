from typing import Any
from torchgeo.datasets import IntersectionDataset
from torchgeo.datasets.utils import BoundingBox

# This wrapper of intersection dataset is only needed until Kornia releases 0.7.4
class KorniaIntersectionDataset(IntersectionDataset):

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # All datasets are guaranteed to have a valid query
        samples = [ds[query] for ds in self.datasets]

        sample = self.collate_fn(samples)

        # Only pass valid Kornia data keys to transforms. This won't be needed after Kornia's next release (>0.7.4).
        kornia_sample = {'image':sample['image'], 'mask':sample['mask']}

        if self.transforms is not None:
            kornia_sample = self.transforms(kornia_sample)

        sample = { 'image' : kornia_sample['image'], 'mask' : kornia_sample['mask'], 'crs': sample['crs'], 'bbox' : sample['bbox'] }

        return sample