from torchgeo.samplers import RandomBatchGeoSampler
import torch
from typing import List, Iterator, Tuple
from torchgeo.samplers.utils import get_random_bounding_box
from torchgeo.datasets.utils import BoundingBox

class TallVegetationSampler(RandomBatchGeoSampler):
    """
    Batch sampler maximizing exposure to tall vegetation by prioritizing 
    scenes with high proportion of tall trees.
    """
    def __init__(
        self,
        dataset,
        patch_size: int,
        batch_size: int,
        length: int,
        height_threshold: float = 15.0,  # Threshold for "tall" vegetation
        oversample_factor: float = 3.0,
        nan_value: float = -32767.0
    ):
        super().__init__(dataset, patch_size, batch_size, length)
        self.height_threshold = height_threshold
        self.oversample_factor = oversample_factor
        self.nan_value = nan_value
        self.dataset = dataset

    def _calculate_tall_vegetation_score(self, heights: torch.Tensor) -> float:
        """Calculate score based on proportion and magnitude of tall vegetation."""
        valid_mask = heights != self.nan_value
        valid_heights = heights[valid_mask]
        
        if len(valid_heights) == 0:
            return 0.0
            
        # Calculate proportion of pixels above threshold
        tall_ratio = (valid_heights >= self.height_threshold).float().mean().item()
        
        # Calculate average height of tall vegetation
        tall_heights = valid_heights[valid_heights >= self.height_threshold]
        if len(tall_heights) > 0:
            tall_mean = tall_heights.mean().item()
        else:
            tall_mean = 0.0
            
        # Score combines both ratio and magnitude
        return tall_ratio * tall_mean

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        for _ in range(len(self)):
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)
            
            n_candidates = int(self.batch_size * self.oversample_factor)
            candidates = []
            
            for _ in range(n_candidates):
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                patch_data = self.dataset.datasets[1][bounding_box]
                
                if 'mask' in patch_data:
                    score = self._calculate_tall_vegetation_score(patch_data['mask'])
                    candidates.append((bounding_box, score))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected_bboxes = [c[0] for c in candidates[:self.batch_size]]
            
            while len(selected_bboxes) < self.batch_size:
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                selected_bboxes.append(bounding_box)
                
            yield selected_bboxes

class ProgressiveGeoSampler(RandomBatchGeoSampler):
    """
    Sampler that progressively relaxes criteria to find diverse patches.
    Prioritizes tall trees in simple scenes, then falls back to other criteria.
    
    Args:
        dataset: The dataset to sample from
        patch_size: Size of patches to sample
        batch_size: Number of patches per batch
        length: Number of batches per epoch
        min_height_threshold: Minimum height to consider for "tall" vegetation
        min_tall_ratio: Minimum ratio of tall vegetation required
        max_attempts: Maximum sampling attempts per batch
        max_complexity: Maximum complexity for "simple" scenes
        min_valid_ratio: Minimum ratio of valid (non-nan) pixels
        nan_value: Value indicating no-data points
    """
    def __init__(
        self,
        dataset,
        patch_size: int,
        batch_size: int,
        length: int,
        min_height_threshold: float = 15.0,
        min_tall_ratio: float = 0.1,
        max_attempts: int = 2000,  # Increased from original
        max_complexity: float = 0.4,
        min_valid_ratio: float = 0.5,
        nan_value: float = -32767.0
    ):
        super().__init__(dataset, patch_size, batch_size, length)
        self.min_height_threshold = min_height_threshold
        self.min_tall_ratio = min_tall_ratio
        self.max_attempts = max_attempts
        self.max_complexity = max_complexity
        self.min_valid_ratio = min_valid_ratio
        self.nan_value = nan_value
        self.dataset = dataset
        
        # Attempt thresholds for different stages
        self.stage_thresholds = {
            'ideal': int(0.4 * max_attempts),     # Try to find ideal patches first
            'tall_only': int(0.7 * max_attempts), # Then focus on just tall trees
            'complex': int(0.9 * max_attempts),   # Then accept complex scenes
            'random': max_attempts                # Finally random sampling
        }
        
    def _calculate_patch_metrics(self, heights: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        Calculate all metrics for a patch.
        Returns (valid_ratio, complexity, tall_ratio, height_score).
        """
        # Calculate valid ratio first
        valid_mask = heights != self.nan_value
        valid_ratio = valid_mask.float().mean().item()
        
        if valid_ratio < self.min_valid_ratio:
            return valid_ratio, 1.0, 0.0, 0.0
            
        valid_heights = heights[valid_mask]
        if len(valid_heights) == 0:
            return valid_ratio, 1.0, 0.0, 0.0
            
        # Calculate basic statistics
        mean = valid_heights.mean()
        if mean == 0:
            return valid_ratio, 1.0, 0.0, 0.0
            
        # Complexity score using CV and skewness
        std = valid_heights.std()
        cv = std / mean
        skewness = torch.mean(((valid_heights - mean) / std)**3)
        complexity = torch.tanh(cv * abs(skewness)) / 2 + 0.5
        
        # Calculate tall tree ratio
        tall_mask = valid_heights >= self.min_height_threshold
        tall_ratio = tall_mask.float().mean().item()
        
        # Overall height score (could be used for additional filtering)
        height_score = tall_ratio * (1 - complexity.item())
        
        return valid_ratio, complexity.item(), tall_ratio, height_score
        
    def _evaluate_patch(self, patch_data: dict, attempts: int) -> Tuple[bool, float]:
        """
        Evaluate if a patch meets current criteria based on sampling stage.
        Returns (accepted, score).
        """
        if 'mask' not in patch_data:
            return False, 0.0
            
        valid_ratio, complexity, tall_ratio, height_score = self._calculate_patch_metrics(patch_data['mask'])
        
        # Basic validity check
        if valid_ratio < self.min_valid_ratio:
            return False, 0.0
            
        # Progressive criteria based on attempt count
        if attempts < self.stage_thresholds['ideal']:
            # Stage 1: Look for simple scenes with tall trees
            if complexity <= self.max_complexity and tall_ratio >= self.min_tall_ratio:
                return True, height_score
                
        elif attempts < self.stage_thresholds['tall_only']:
            # Stage 2: Accept any scene with sufficient tall trees
            if tall_ratio >= self.min_tall_ratio:
                return True, tall_ratio
                
        elif attempts < self.stage_thresholds['complex']:
            # Stage 3: Accept complex scenes to ensure model learns challenging cases
            if complexity > self.max_complexity:
                return True, complexity
                
        else:
            # Stage 4: Accept any valid patch, preferring those with some height variation
            return True, max(0.1, height_score)
            
        return False, 0.0

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Iterator that yields batches using progressive criteria."""
        for _ in range(len(self)):
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)
            
            selected_bboxes = []
            attempts = 0
            
            while len(selected_bboxes) < self.batch_size and attempts < self.max_attempts:
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                
                patch_data = self.dataset.datasets[1][bounding_box]
                accepted, score = self._evaluate_patch(patch_data, attempts)
                
                if accepted:
                    selected_bboxes.append(bounding_box)
                
                attempts += 1
            
            # Fill any remaining spots with random patches
            while len(selected_bboxes) < self.batch_size:
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                selected_bboxes.append(bounding_box)
                
            yield selected_bboxes

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
        max_attempts: int = 640,
        nan_value: float = -32767.0,
        initial_max_nodata_ratio: float = 0.4,
        initial_max_low_veg_ratio: float = 0.4
    ):
        super().__init__(dataset, patch_size, batch_size, length)
        self.max_diversity_threshold = max_diversity_threshold
        self.min_diversity_threshold = min_diversity_threshold
        self.max_attempts = max_attempts
        self.nan_value = nan_value
        self.dataset = dataset
        self.initial_max_nodata_ratio = initial_max_nodata_ratio
        self.initial_max_low_veg_ratio = initial_max_low_veg_ratio

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
