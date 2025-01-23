from torchgeo.samplers import RandomBatchGeoSampler
import torch
from typing import List, Iterator
from torchgeo.samplers.utils import get_random_bounding_box
from torchgeo.datasets.utils import BoundingBox

class HeightDiversityBatchSampler(RandomBatchGeoSampler):
    """
    Batch sampler that prioritizes patches with diverse height distributions.
    Uses statistical moments to assess diversity efficiently.
    
    Args:
        dataset: The dataset to sample from
        patch_size: Size of patches to sample
        batch_size: Number of patches per batch
        diversity_threshold: Minimum diversity score to accept patch
        max_attempts: Maximum attempts to find diverse patch
        nan_value: Value indicating no-data points
    """
    def __init__(
        self,
        dataset,
        patch_size: int,
        batch_size: int,
        length: int,
     #   generator: Generator | None = None, # for compatibility with torchgeo 0.7    
        max_diversity_threshold: float = 10.0,
        min_diversity_threshold: float = 6.0,
        max_attempts: int = 640,
        nan_value: float = -32767.0
    ):
        super().__init__(dataset, patch_size, batch_size, length)
        self.max_diversity_threshold = max_diversity_threshold
        self.min_diversity_threshold = min_diversity_threshold
        self.max_attempts = max_attempts
        self.nan_value = nan_value
        self.dataset = dataset
        
    def _calculate_diversity_score(self, heights: torch.Tensor) -> float:
        """
        Calculate diversity score using statistical moments.
        Higher scores indicate better height diversity.
        """
        # Get valid heights
        valid_mask = heights != self.nan_value
        valid_heights = heights[valid_mask]
        
        if len(valid_heights) == 0:
            return 0.0
            
        # Calculate basic statistics
        mean = valid_heights.mean()
        std = valid_heights.std()
        height_range = valid_heights.max() - valid_heights.min()
        
        # Get interquartile range
        q75, q25 = torch.quantile(valid_heights, torch.tensor([0.75, 0.25]))
        iqr = q75 - q25
        
        # Combine metrics into single score
        # Using geometric mean to balance influence of each metric
        # Adding small epsilon to avoid zero scores
        eps = 1e-6
        diversity_score = (std * height_range * (iqr + eps)) ** (1/3)
        #print(f'Diversity score: {diversity_score}, item: {diversity_score.item()}, max height: {valid_heights.max()}, min height : {valid_heights.min()}, average: {mean}')
        return diversity_score.item()


    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """
        Iterator that yields batches of diverse patches.
        Keeps good patches that meet threshold and continues searching for remaining spots.
        """        
        for _ in range(len(self)):

            # Choose initial random batch
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)
            
            # Initialize lists to store our best bboxes and their scores
            selected_bboxes = []
            attempts = 0

            while len(selected_bboxes) < self.batch_size and attempts < self.max_attempts:
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)

                patch_data = self.dataset.datasets[1][bounding_box]
                if 'mask' in patch_data:
                    score = self._calculate_diversity_score(patch_data['mask'])
                else:
                    score = 0.0

                progress = float(attempts)/float(self.max_attempts)
                diversity_threshold = self.max_diversity_threshold
                if progress > 0.25  and len(selected_bboxes) <= float(self.batch_size)*0.25:
                    diversity_threshold = self.min_diversity_threshold + (self.max_diversity_threshold-self.min_diversity_threshold)*(1.0 - 0.25)
                if progress > 0.5 and len(selected_bboxes) <= float(self.batch_size)*0.5:
                    diversity_threshold = self.min_diversity_threshold + (self.max_diversity_threshold-self.min_diversity_threshold)*(1.0 - 0.5)
                if progress > 0.75 and len(selected_bboxes) <= float(self.batch_size)*0.75:
                    diversity_threshold = self.min_diversity_threshold
                if progress > 0.90 and len(selected_bboxes) <= float(self.batch_size):
                    diversity_threshold = 3.0
                    
                if score >= diversity_threshold: 
                    selected_bboxes.append(bounding_box)
                attempts += 1
            
            # print(f'Selected patches based on score: {len(selected_bboxes)}')
            # If we still haven't filled the batch, fill with random patches
            if len(selected_bboxes) < self.batch_size:
                remaining = self.batch_size - len(selected_bboxes)
                for _ in range(remaining):
                    bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                    selected_bboxes.append(bounding_box)
                
            assert len(selected_bboxes) == self.batch_size, f"Invalid batch size: {len(selected_bboxes)}"    
            yield selected_bboxes