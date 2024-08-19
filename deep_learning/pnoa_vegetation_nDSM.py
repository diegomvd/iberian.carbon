from torchgeo.datasets import RasterDataset

class PNOAnDSMV(RasterDataset):
    """Spanish vegetation Normalized Surface Digital Model

    """
    is_image = False 
    filename_glob = "PNOA_*"

    filename_regex = r'PNOA_(?P<date>\d{4})'
    date_format = "%Y"



 
