from torchgeo.samplers import RandomBatchGeoSampler
import torch
from typing import List, Iterator, Tuple
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
        max_diversity_threshold: float = 8.0,
        min_diversity_threshold: float = 4.0,
        # tall_batch_ratio: float = 0.4,    
        # tall_tree_ratio: float = 0.25,
        # tall_height_threshold: float = 10.0,
        max_attempts: int = 1200,
        nan_value: float = -32767.0,
        initial_max_nodata_ratio: float = 0.4,
        initial_max_low_veg_ratio: float = 0.4
    ):
        super().__init__(dataset, patch_size, batch_size, length)
        self.max_diversity_threshold = max_diversity_threshold
        self.min_diversity_threshold = min_diversity_threshold
        # self.tall_tree_ratio = tall_tree_ratio
        # self.tall_height_threshold = tall_height_threshold
        self.max_attempts = max_attempts
        self.nan_value = nan_value
        self.dataset = dataset
        self.initial_max_nodata_ratio = initial_max_nodata_ratio
        self.initial_max_low_veg_ratio = initial_max_low_veg_ratio

        # self.tall_batch_size = int(batch_size*tall_batch_ratio)
        # self.general_batch_size = batch_size - self.tall_batch_size

    # def _evaluate_patch_tall_trees(self, heights: torch.Tensor) -> bool:
    #     """Check if patch contains sufficient tall trees."""
    #     valid_mask = heights != self.nan_value
    #     valid_heights = heights[valid_mask]
    #     if len(valid_heights) == 0:
    #         return False
    #     tall_ratio = (valid_heights >= self.tall_height_threshold).float().mean().item()
    #     return tall_ratio >= self.tall_tree_ratio  # Minimum ratio of tall trees

    def _print_patch_stats(self, heights: torch.Tensor, score: float, progress: float) -> None:
        """Print detailed patch statistics ignoring nan values."""            
        # Get valid heights (non-nan)
        valid_mask = heights != self.nan_value
        valid_heights = heights[valid_mask]
        
        if len(valid_heights) == 0:
            print("Empty patch (all nan values)")
            return
            
        # Calculate statistics on valid data
        stats = {
            "min_height": valid_heights.min().item(),
            "max_height": valid_heights.max().item(),
            "mean_height": valid_heights.mean().item(),
            "median_height": torch.median(valid_heights).item(),
            "std_height": valid_heights.std().item(),
            "diversity_score": score,
            "valid_data_ratio": valid_mask.float().mean().item(),
            "low_veg_ratio": ((valid_heights >= 0) & (valid_heights <= 1)).float().mean().item(),
        }
        
        # Height distribution in meaningful bins
        bins = torch.tensor([0, 1, 2, 4, 8, 12, 16, 20, 25, float('inf')])
        hist = torch.histogram(valid_heights, bins=bins)
        height_dist = {f"{bins[i].item():.1f}-{bins[i+1].item():.1f}m": 
                      f"{(hist.hist[i].item()/len(valid_heights)*100):.1f}%" 
                      for i in range(len(hist.hist))}
        
        print(f"\nPatch Statistics (progress: {progress:.2f}):")
        print(f"Basic Stats:")
        print(f"  Valid Data: {stats['valid_data_ratio']*100:.1f}%")
        print(f"  Low Veg (0-1m): {stats['low_veg_ratio']*100:.1f}%")
        print(f"  Height Range: {stats['min_height']:.1f}m - {stats['max_height']:.1f}m")
        print(f"  Mean Height: {stats['mean_height']:.1f}m")
        print(f"  Median Height: {stats['median_height']:.1f}m")
        print(f"  Height StdDev: {stats['std_height']:.1f}m")
        print(f"  Diversity Score: {stats['diversity_score']:.1f}")
        
        print("Height Distribution:")
        for range_str, percentage in height_dist.items():
            print(f"  {range_str}: {percentage}")
        print("-" * 50)

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

    def _check_patch_constraints(self, heights: torch.Tensor, progress: float) -> bool:
        """Check nodata and low vegetation constraints with progressive relaxation"""
        valid_mask = heights != self.nan_value
        valid_ratio = valid_mask.float().mean()
        
        # Progressive relaxation of constraints
        max_nodata_ratio = self.initial_max_nodata_ratio
        max_low_veg_ratio = self.initial_max_low_veg_ratio
        
        if progress > 0.5:
            max_nodata_ratio *= 1.5
            max_low_veg_ratio *= 1.5
        if progress > 0.75:
            max_nodata_ratio *= 2
            max_low_veg_ratio *= 2
        if progress > 0.9:
            return True  # Accept any patch near the end
            
        if valid_ratio < (1 - max_nodata_ratio):
            return False
            
        valid_heights = heights[valid_mask]
        if len(valid_heights) == 0:
            return False
            
        low_veg_ratio = ((valid_heights >= 0) & (valid_heights <= 1.5)).float().mean()
        return low_veg_ratio <= max_low_veg_ratio

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
                progress = float(attempts)/float(self.max_attempts)

                patch_data = self.dataset.datasets[1][bounding_box]
                if 'mask' not in patch_data:
                    attempts += 1
                    continue
                
                if not self._check_patch_constraints(patch_data['mask'], progress):
                    attempts += 1
                    continue
                # print(f'Found a patch respecting constraints after {attempts} attempts')
                score = self._calculate_diversity_score(patch_data['mask'])

                # if attempts % 1 == 0:  # Print every 50 attempts
                #     self._print_patch_stats(patch_data['mask'], score, progress)
            
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
