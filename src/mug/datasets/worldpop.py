'''
 (C) Copyright IBM Corp. 2024.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
           http://www.apache.org/licenses/LICENSE-2.0
     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
 Project name: Model Urban Growth MUG
'''

"""WorldPop datasets."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
from rasterio.crs import CRS

from torch.utils.data import Dataset
from torchgeo.datasets import BoundingBox, RasterDataset
from torchgeo.datasets.utils import download_url

from . import SpatiotemporalRasterDataset
from .utils import concat_samples
from ..transforms import (
    normalize_binary_land_mask,
    factor,
)

# Age structures

AGE_0 = 0
AGE_1 = 1
AGE_5 = 5
AGE_10 = 10
AGE_15 = 15
AGE_20 = 20
AGE_25 = 25
AGE_30 = 30
AGE_35 = 35
AGE_40 = 40
AGE_45 = 45
AGE_50 = 50
AGE_55 = 55
AGE_60 = 60
AGE_65 = 65
AGE_70 = 70
AGE_75 = 75
AGE_80 = 80

# Sex categories
SEX_FEMALE = "f"
SEX_MALE = "m"

# ESA CCI-LC categories

LC_CROPLAND = 11
LC_VEGETATION = 40
LC_GRASSLAND = 130
LC_LICHENS = 140
LC_SPARSE_VEGETATION = 150
LC_FLOODED = 160
LC_URBAN = 190
LC_BARE = 200

MAXIMUM_YEAR_DATASET = 2005

class _StaticDataset(RasterDataset):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
        cmap: str = "gray"
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)
            cmap: a valid Matplotlib colormap name

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.download = download
        self.checksum = checksum
        self.cmap = cmap

        self._verify()

        super().__init__(root, crs, res, transforms=transforms, cache=cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "**", self.filename_glob)
        filelist = [fname for fname in glob.iglob(pathname, recursive=True)]
        if len(filelist) > 0:
            return
        
        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`,"
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url, self.root, md5=self.md5s if self.checksum else None
        )

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample["mask"].squeeze().numpy()
        ncols = 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        axs.imshow(mask, cmap=self.cmap)
        axs.axis("off")
        if show_titles:
            axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return

class _DynamicDataset(SpatiotemporalRasterDataset):
    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "**", self.filename_glob)
        filelist = [fname for fname in glob.iglob(pathname, recursive=True)]
        if len(filelist) > MAXIMUM_YEAR_DATASET - min(self.yearlist):
            return
        
        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`,"
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
    
    def _download():
        raise NotImplementedError("SpatiotemporalRasterDataset is an abstract class.")

class AgeSex(_DynamicDataset):
    """WorldPop Age/Sex structures dataset.

    The `WorldPop
    <https://www.worldpop.org/>`__ Age/Sex structures dataset provides rasterized,
    geo-referenced, binary maps of age and sex structures with a spatial
    resolution of 30 arc seconds for the entire globe.
    
    If you use this dataset in your research, please cite the following work:

    * https://doi.org/10.1080/20964471.2019.1625151
    """  # noqa: E501

    date_format = "%Y"
    separate_files = False

    url = "ftp://ftp.worldpop.org.uk/GIS/AgeSex_structures/Global_2000_2020/{}/0_Mosaicked/global_mosaic_1km/global_{}_{}_{}_1km.tif"  # noqa: E501

    # TODO: add MD5 checksums
    md5s = {
        SEX_FEMALE: {
            AGE_0: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_1: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_5: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_10: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_15: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_20: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_25: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_30: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_35: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_40: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_45: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_50: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_55: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_60: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_65: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_70: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_75: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_80: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ]
        },
        SEX_MALE: {
            AGE_0: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_1: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_5: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_10: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_15: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_20: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_25: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_30: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_35: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_40: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_45: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_50: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_55: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_60: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_65: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_70: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_75: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ],
            AGE_80: [
                (2000, ""),
                (2001, ""),
                (2002, ""),
                (2003, ""),
                (2004, ""),
                (2005, ""),
                (2006, ""),
                (2007, ""),
                (2008, ""),
                (2009, ""),
                (2010, ""),
                (2011, ""),
                (2012, ""),
                (2013, ""),
                (2014, ""),
                (2015, "")
            ]
        }
    }

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
        cmap: str = "gray",
        age: int = AGE_20,
        sex: str = SEX_FEMALE
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)
            cmap: a valid Matplotlib colormap string
            age: one of valid age buckets (0, 1, 5, 10, 15, 20, 25, ..., 80)
            sex: either SEX_FEMALE or SEX_MALE

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.download = download
        self.checksum = checksum
        self.cmap = cmap
        self.age = age
        self.sex = sex
        # dynamically set filename glob, regex according to land cover class
        self.filename_glob = f"global_{self.sex}_{self.age}_*_1km.tif"
        self.filename_regex = fr"^.{{6}}_{self.sex}_{self.age}_(?P<date>\d+)_1km.tif$"
        # imply list of years from list of file MD5 checksums
        self.yearlist = [year for year, _ in self.md5s[self.sex][self.age]]

        self._verify()

        super().__init__(root=root,
                         crs=crs,
                         res=res,
                         transforms=transforms,
                         cache=cache)

    def __str__(self) -> str:
        repr = super().__str__()
        repr += f"\n    sex: {self.sex}"
        repr += f"\n    age: {self.age}"
        return repr

    def _download(self) -> None:
        """Download the dataset."""
        for year, md5 in self.md5s[self.sex][self.age]:
            download_url(
                self.url.format(year, self.sex, self.age, year), 
                self.root,
                md5=md5 if self.checksum else None
            )


class Elevation(_StaticDataset):
    """WorldPop Elevation dataset.

    The `WorldPop
    <https://www.worldpop.org/>`__ Elevation dataset provides a raster,
    geo-referenced, elevation map (in meters) with a spatial resolution
    of 3 arc seconds for the entire globe.

    If you use this dataset in your research, please cite the following work:

    * https://doi.org/10.1080/20964471.2019.1625151
    """  # noqa: E501

    filename_glob = "topo100m.tif"
    filename_regex = r"^.{12}"
    separate_files = False

    url = "ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/"\
        "0_Mosaicked/Elevation/topo100m.tif"  # noqa: E501

    md5s = "a1b17ce0afa630cdbb573b1414a0b178"


class LandCover(_DynamicDataset):
    """WorldPop Elevation dataset.

    The `WorldPop
    <https://www.worldpop.org/>`__ Land Cover dataset provides rasterized,
    geo-referenced, binary maps of several land cover classes processed from the
    ESA CCI-LC dataset with a spatial resolution of 3 arc seconds
    for the entire globe. The following ESA CCI-LC classes are available, as
    described in doi:10.3390/rs12101545:

         11 - Includes original classes 10-30
         40 - Includes original classes 40-120
        130 - Grassland
        140 - Lichens and mosses
        150 - Sparse vegetation
        160 - Includes original classes 160-180
        190 - Urban area
        200 - Bare area.

    If you use this dataset in your research, please cite the following work:

    * https://doi.org/10.1080/20964471.2019.1625151
    """  # noqa: E501

    date_format = "%Y"
    separate_files = False

    url = "ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/"\
        "0_Mosaicked/ESA_CCI_Annual/ESA_CCI_{}/Class{}/"\
        "BinaryMosaic_1_0_NoData/"\
        "ESACCI_LC_Reclassified_yr{}_cls{}_1_0_NoData.tif"  # noqa: E501

    md5s = {
        LC_CROPLAND: [
            (2000, "2a578f4035402ca75749a6c18bcd2507"),
            (2001, "03a05930abcd41b25a163826d8abfd1b"),
            (2002, "dad21a2d6ac9f27b8a1b5f03a63a9b59"),
            (2003, "e0c7029c221376f51b7030dcbf030784"),
            (2004, "1f04f43bae47625cf6d8be60e4b6171c"),
            (2005, "ea9c56f644acbcd05e55de3f1cd7546a"),
            (2006, "916a1f8745fc536b89e1a936063a45e9"),
            (2007, "ef29398b06664b023f231285cf10412a"),
            (2008, "e55cd5e7eac51d64590b36cea3cb4578"),
            (2009, "52b4f047f1a094561f02ebc8e12f444a"),
            (2010, "84f9de8d4ec1c0d396b11b4564309e9a"),
            (2011, "ee8f60ea7d4711fe1fe222385601b85b"),
            (2012, "2eb4b7451a868699292306db0d2102be"),
            (2013, "5db6a6227635c00bf43cf3d4d5e8fe91"),
            (2014, "892a1174d204691c111bc99e6860d6d5"),
            (2015, "191b7211c5c229da939182a834775f88")
        ],
        LC_VEGETATION: [
            (2000, "be970f2ac9ce0d4856fa0f06a370be9b"),
            (2001, "736e98711f44394cac1a52ad599b4899"),
            (2002, "2275f54f212ec67fda417fc710cfa380"),
            (2003, "a264c7af0673cf02a3c576fb9b531038"),
            (2004, "5f14c8bc5f0525b248016f23dc7b2adb"),
            (2005, "188f56b636085f31698a4f2023f4bcd9"),
            (2006, "a8c9f199986478e31c52b6d6df492ee7"),
            (2007, "85c5c1a7b7f6fe2107a713ffc3ea5ff3"),
            (2008, "93e4df75ed1eb5b794651b684cc86439"),
            (2009, "4326638b37d20d6fbbd1ed4b158d8522"),
            (2010, "ceefad43e6d00e71fbfec5f5c09aab5e"),
            (2011, "196f09856786e93fb67b5dc5abe61f3e"),
            (2012, "26931dd42d2bc9777cc9a3e70f289d45"),
            (2013, "715e1e44ed1e68da3c507a5304e6a49f"),
            (2014, "d539571569b2f8aa2c2fda9bf8e4749a"),
            (2015, "95b7e6d7a56a71d85f9558e701cb900e")
        ],
        LC_GRASSLAND: [
            (2000, "a61a1f70829c077d9d0f2b070501c185"),
            (2001, "9766d07fc833548829568bdf9d096cca"),
            (2002, "7e83134df42c3c9f20d6964941ec0a51"),
            (2003, "417366eb604e01d89f926ba8793f1eee"),
            (2004, "f48d94929035a129357a6829f974da8e"),
            (2005, "b3732c1e61a6a5e7117929afe6f5078d"),
            (2006, "ae31de0b7b5a1eb15364fff45da2453d"),
            (2007, "333343f6fd1b737b8148ce11a0910c95"),
            (2008, "8bf67427c4db5e8857905cc98cdcdd47"),
            (2009, "3b728acf7cf0f89e4e961d7fbed558b6"),
            (2010, "062b71c35c386c119159b4d08f2f335a"),
            (2011, "2599e6da8dd423919738729d104b6cd6"),
            (2012, "00a4495cb8abe0e0120bf36ee3c8b5a1"),
            (2013, "6db7e35a5fd2b0f4f099dc2f8dbc3c3b"),
            (2014, "7a7896df62b4311cfb89684f7c22c9f5"),
            (2015, "9f9bd4ca7c358c2e032863b78ea44bf6")
        ],
        LC_LICHENS: [
            (2000, "d5d6924be9395b360e1888620d6aee4c"),
            (2001, "b75e0739dbe2d8173eafa362aed173d8"),
            (2002, "3d2e55d79c55eed298d855d306aa34ea"),
            (2003, "c1ce2791362ea3370fb1673a00eeb5da"),
            (2004, "14b277a4d3ef8ff683c621300c7723d7"),
            (2005, "f95131e0fa0e10e31ada4c336c2df066"),
            (2006, "186692b1490c7a119e9d5aea70d94578"),
            (2007, "8f1d0963d32344dbb09d4b1c80c78915"),
            (2008, "dec2c3bbe0eac1af742ba7c67469e495"),
            (2009, "81146dbbb46f279099e34875d9fb5fdf"),
            (2010, "bcca81db6e642171c06aa7db4faef7e3"),
            (2011, "1745c3c0b82941a3b7eca3e5b2e49de9"),
            (2012, "9aad3590ecdd73730b0f0a0605fca7c0"),
            (2013, "eed953d2ce8e7f7021cbfe8381c3fab6"),
            (2014, "183cd325524d5f1973c4c66461d7082b"),
            (2015, "bc0f0593003ec1f68bbf38059168868f")
        ],
        LC_SPARSE_VEGETATION: [
            (2000, "16720f7bfbffa5f69e7dafc4222d1190"),
            (2001, "9a6125504796cffac2a843efa0c72bc8"),
            (2002, "17e55670df24e5907ffef3211791b0ca"),
            (2003, "e2161f97ee36defeedeb3692e7502ca4"),
            (2004, "5258d9d9fa6e96448b09501baf60147e"),
            (2005, "9095e0658c920e19eca7703f305d9185"),
            (2006, "60f489b6e8511111d639a64bc25f724a"),
            (2007, "b81109dfc499938bbd68b5a0df5f6500"),
            (2008, "8bb20f746bad93300abc17a1bfbf4d75"),
            (2009, "f469f8daf76207c1ce35878d0db5447a"),
            (2010, "686e13626302bc380ae06137c6825ac9"),
            (2011, "f076fb9eebbd080a77f1f061d28c80aa"),
            (2012, "b3e39e264c186ce6fa90d22f8e685560"),
            (2013, "50edf16f4d7b16b05c42b6d8d5fc701f"),
            (2014, "b8df2d7c680a2295fc9db7bd8f755955"),
            (2015, "6e464a644a06a4fff9062abdab12a05d")
        ],
        LC_FLOODED: [
            (2000, "bea83beb172c21437908ee2a4b935112"),
            (2001, "335f25a48e1da245f310969539bcb409"),
            (2002, "093114b192dd2e51d3b0267f1dc7e35b"),
            (2003, "f693ab88b40b159a4a9c26e33ae8efa6"),
            (2004, "290c3b1aa97c8f443c0f7a21d063e4be"),
            (2005, "cdc04da5aafbfceaaf4ddae14ba354b4"),
            (2006, "fe444486499c0d78ff9fcd26ace84e0b"),
            (2007, "6df56c5210bb028673aa86d546047a37"),
            (2008, "f8cf0744c2afa34fb8ecddf330f0a588"),
            (2009, "b75a984f2337e7e14a62c38bbb203496"),
            (2010, "14e4082919369a0d566e7e866542521e"),
            (2011, "a952bc1eebec2b89c681b8d779a4f6a7"),
            (2012, "f7af27b8e59a507602c5607e13cd4749"),
            (2013, "8b41892acc4e8f988d4ab583f434b617"),
            (2014, "a3195ca49a5145d11522375bcc56f340"),
            (2015, "b76682817986c3dd286025d78e281043")
        ],
        LC_URBAN: [
            (2000, "2b81e039ebdd65f3d76a74a7f547dac7"),
            (2001, "43a50fdf3d5ebbb4e0e269e28857f6ac"),
            (2002, "6708120877281b4ad2262726d3c4d53f"),
            (2003, "81fb392a9b6b4b165273c302cd212076"),
            (2004, "89d30b94dd3165a14ba2bb20294cc0f9"),
            (2005, "8feb0aa65c3b3c9ee7f1169bf5e18900"),
            (2006, "bd99c8fccaff34dfb06a558c93956420"),
            (2007, "599ceb72156f15556554d980b893acf4"),
            (2008, "74b99e28b9722771e4592ef2d14356fc"),
            (2009, "b3a70d103f585e154f2e21578ec6749e"),
            (2010, "2891d4d8134abb5a4e8e5f0b7db4914d"),
            (2011, "eaf793f1b121199b5d06ba1860de6e31"),
            (2012, "0a0e91c0351a10b7aa05d6da17b7e302"),
            (2013, "e4b07f25597a2e319ed52aece771d164"),
            (2014, "5adfc63b150973093f01e0770e901b50"),
            (2015, "e6566aec727dfbb7434a36f1133be9cf")
        ],
        LC_BARE: [
            (2000, "9b30d5c14c00df511a7fb21606fcbaab"),
            (2001, "07115c5e1006f75374ed8ba5e0cf93c0"),
            (2002, "bafec78690414b8831c36eeeda352f74"),
            (2003, "22d480d28306c541be43d143d85a3a98"),
            (2004, "ffc1a9ef38cb89b04521b0e253504d8f"),
            (2005, "e3dcdb89f1ef2961fadad36aca4a1932"),
            (2006, "563cf59f39772142b4fbc3e6b2fa92d1"),
            (2007, "d47a7581aef37138224721032bac0ae8"),
            (2008, "f8b53faa2d4595b4087884c92ba5a094"),
            (2009, "1fdd09e3843a17c4ae267a80f20bd427"),
            (2010, "4eb0f64835efc30e16ccb7c62749e118"),
            (2011, "d92759d5e8b52b891c403b770498fce2"),
            (2012, "10b6c6d969a1db9e9dc59ffeff4778f0"),
            (2013, "5c348af5132849d9ed174b426d225bb4"),
            (2014, "a2c2a2a065548944340d22d5f039d7cb"),
            (2015, "989afa36f6e4801f342828dd015f93e8")
        ]
    }

    def __init__(
        self,
        category: int,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)
            cmap: a valid Matplotlib colormap string
            category: one of the valid ESA CCI-LC classes 
                (11, 40, 130, 140, 150, 160, 190, 200)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.download = download
        self.checksum = checksum
        self.category = category
        # dynamically set filename glob, regex according to land cover class
        self.filename_glob = f"ESACCI_LC_Reclassified_yr*_cls{self.category}_1_0_NoData.tif"
        self.filename_regex = fr"^.{{25}}(?P<date>\d+)_cls{self.category}_.*$"
        # imply list of years from list of file MD5 checksums
        self.yearlist = [year for year, _ in self.md5s[self.category]]

        self._verify()

        super().__init__(root=root, crs=crs, res=res, transforms=transforms, cache=cache)

    def _download(self) -> None:
        """Download the dataset."""
        for year, md5 in self.md5s[self.category]:
            download_url(
                self.url.format(year, self.category, year, self.category), 
                self.root,
                md5=md5 if self.checksum else None
            )


class CropLandCover(LandCover):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = normalize_binary_land_mask,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(LC_CROPLAND, root, crs, res, transforms, cache, download, checksum)


class VegetationLandCover(LandCover):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = normalize_binary_land_mask,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(LC_VEGETATION, root, crs, res, transforms, cache, download, checksum)


class GrassLandCover(LandCover):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = normalize_binary_land_mask,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(LC_GRASSLAND, root, crs, res, transforms, cache, download, checksum)


class LichensLandCover(LandCover):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = normalize_binary_land_mask,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(LC_LICHENS, root, crs, res, transforms, cache, download, checksum)


class SparseVegetationLandCover(LandCover):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = normalize_binary_land_mask,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(LC_SPARSE_VEGETATION, root, crs, res, transforms, cache, download, checksum)


class FloodedLandCover(LandCover):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = normalize_binary_land_mask,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(LC_FLOODED, root, crs, res, transforms, cache, download, checksum)


class UrbanLandCover(LandCover):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = normalize_binary_land_mask,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(LC_URBAN, root, crs, res, transforms, cache, download, checksum)


class BareLandCover(LandCover):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = normalize_binary_land_mask,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(LC_BARE, root, crs, res, transforms, cache, download, checksum)


class PixelArea(_StaticDataset):
    """WorldPop Pixel Area dataset.

    The `WorldPop
    <https://www.worldpop.org/>`__ Pixel Area dataset provides pixel area,
    with a spatial resolution of 30 arc seconds for the entire globe.

    If you use this dataset in your research, please cite the following work:

    * https://doi.org/10.1080/20964471.2019.1625151
    """  # noqa: E501
    
    filename_glob = "global_px_area_1km.tif"
    filename_regex = r"^.{22}"
    separate_files = False

    url = "ftp://ftp.worldpop.org.uk/GIS/Pixel_area/Global_2000_2020/"\
        "0_Mosaicked/global_px_area_1km.tif"  # noqa: E501

    md5s = "302f2429eef66fe147c943feca273f02"


class Population(_DynamicDataset):
    """WorldPop Population dataset.

    The `WorldPop
    <https://www.worldpop.org/>`__ Population dataset provides rasterized,
    geo-referenced, population counts with a spatial resolution of
    30 arc seconds for the entire globe.

    If you use this dataset in your research, please cite the following work:

    * https://doi.org/10.1080/20964471.2019.1625151
    """  # noqa: E501
    
    filename_glob = "ppp_*_1km_Aggregated.tif"
    filename_regex = r"^.{4}(?P<date>\d+)_1km_.*$"
    date_format = "%Y"
    separate_files = False

    url = "ftp://ftp.worldpop.org.uk/GIS/Population/Global_2000_2020/{}/0_Mosaicked/ppp_{}_1km_Aggregated.tif"  # noqa: E501

    md5s = {   
        (2000, "5e94c682f275685e8428dd413e21cd59"),
        (2001, "a724d78b7fbf7df389fb52aa6a6169a2"),
        (2002, "e79b0541c17eabea765d04e67386e29a"),
        (2003, "c7c0d2963b1d88588134ff00a3f826f1"),
        (2004, "640d8ac7a1db7f87ecf15133ed8213a2"),
        (2005, "8fe15114ef364a749e542ff8e0ffa934"),
        (2006, "2db2aa1b434b218e567513c87c59d95e"),
        (2007, "0d60bb62347075dbee774d821912bd03"),
        (2008, "adeda6e67c4a1233653926c42cdd9941"),
        (2009, "ecb960a3fcfb62100034b8abc4ecbfa7"),
        (2010, "ab2798b98b18efac4165e80bdc52eb40"),
        (2011, "13b1b317ee47a9049dcfcd544d3571c5"),
        (2012, "b9dac76d48054fbe2b6d584114c6e32d"),
        (2013, "32214916a16b87b8c96df61cd9303872"),
        (2014, "d7d79a6908b6bbeaeef78c6e925f34e0"),
        (2015, "a1167ec811d902677267e9d93bbebeff")
    }

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
        cmap: str = "gray",
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)
            cmap: a valid Matplotlib colormap string

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.download = download
        self.checksum = checksum
        self.cmap = cmap
        # imply list of years from list of file MD5 checksums
        self.yearlist = [year for year, _ in self.md5s]

        self._verify()

        super().__init__(root=root, crs=crs, res=res, transforms=transforms, cache=cache)

    def _download(self) -> None:
        """Download the dataset."""
        for year, md5 in self.md5s:
            download_url(
                self.url.format(year, year), self.root, md5=md5 if self.checksum else None
            )


class ProtectedAreas(_DynamicDataset):
    """WorldPop Protected Areas Category 1 dataset.

    The `WorldPop
    <https://www.worldpop.org/>`__ Protected Areas Category 1 dataset
    provides rasterized, geo-referenced, population counts with a
    spatial resolution of 3 arc seconds for the entire globe.

    If you use this dataset in your research, please cite the following work:

    * https://doi.org/10.1080/20964471.2019.1625151
    """  # noqa: E501
    filename_glob = "WDPA_pre*_100m_1_0_NoData_CAT1.tif"
    filename_regex = r"^.{8}(?P<date>\d+)_100m_*.tif$"
    date_format = "%Y"
    separate_files = False

    url = "ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/WDPA/WDPA_1/WDPA_{}_CAT1/WDPA_{}_CAT1_1_0_ND/WDPA_pre{}_100m_1_0_NoData_CAT1.tif"  # noqa: E501

    md5s = {   
        (2000, "2358560cb55546d17ee6e008122cae9b"),
        (2001, "895a19285827405fec47ad45e0d50156"),
        (2002, "31662d9914e20f6fecc0dcf93d06a582"),
        (2003, "9f1df1a9c34fbb8e7ffb09e4b29bdd81"),
        (2004, "47bac7c371cfe08b7505abbbf4925f69"),
        (2005, "1e2788ffb03a15a4dc2152d3fd1022fb"),
        (2006, "102f77369bde6b2ef990bc831f17fd2f"),
        (2007, "07d8dbcf43a86840c5dc9eb58e65859d"),
        (2008, "0fa4ccd8cfd65bccbddcea796117cdf3"),
        (2009, "39a5d72c5120db58df9ead89274dd2a1"),
        (2010, "e26b607b156f262da158762a8229c569"),
        (2011, "e3006a1bfeacfc0cceb7595732776a6d"),
        (2012, "d350eb10edd1619a4c43d8af515cf5da"),
        (2013, "cfb5f84e6d0f803f9b988594a2674deb"),
        (2014, "61a99322265a4a414af48f471a5c26c7"),
        (2015, "d4f1011c221c5aa48a90c86b05685391")
    }

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
        cmap: str = "gray",
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)
            cmap: a valid Matplotlib colormap string

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.download = download
        self.checksum = checksum
        self.cmap = cmap
        # imply list of years from list of file MD5 checksums
        self.yearlist = [year for year, _ in self.md5s]

        self._verify()

        super().__init__(root=root, crs=crs, res=res, transforms=transforms, cache=cache)

    def _download(self) -> None:
        """Download the dataset."""
        for year, md5 in self.md5s:
            download_url(
                self.url.format(year, year, year), self.root, md5=md5 if self.checksum else None
            )

class Roads(_StaticDataset):
    """WorldPop Roads dataset.

    The `WorldPop
    <https://www.worldpop.org/>`__ Roads dataset provides rasterized,
    geo-referenced, binary maps of roads processed from the
    Open Street Maps dataset with a spatial resolution of 3 arc seconds
    for the entire globe.

    If you use this dataset in your research, please cite the following work:

    * https://doi.org/10.1080/20964471.2019.1625151
    """  # noqa: E501
    
    filename_glob = "osmhighway100m8-1710nd.tif"
    filename_regex = r"^.{26}"
    separate_files = False

    url = "ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/OSM_Roads/BinaryMosaic_1_0_NoData/osmhighway100m8-1710nd.tif"  # noqa: E501

    md5s = "a7bbd4037a1d44ddad76843f54a46c0b"


class Slope(_StaticDataset):
    """WorldPop Slope dataset.

    The `WorldPop
    <https://www.worldpop.org/>`__ Slope dataset provides a raster,
    geo-referenced, slope map (in deg) with a spatial resolution
    of 3 arc seconds for the entire globe.

    If you use this dataset in your research, please cite the following work:

    * https://doi.org/10.1080/20964471.2019.1625151
    """  # noqa: E501

    filename_glob = "slope100m.tif"
    filename_regex = r"^.{13}"
    separate_files = False

    url = "ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/Slope/slope100m.tif"  # noqa: E501

    md5s = "d258a6a8a39f75b5e769cef63ad65801"


class Water(_StaticDataset):
    """WorldPop land water dataset.

    The `WorldPop
    <https://www.worldpop.org/>`__ land water dataset provides a raster,
    geo-referenced, binary mask processed from the
    Open Street Maps dataset with a spatial resolution
    of 3 arc seconds for the entire globe.

    If you use this dataset in your research, please cite the following work:

    * https://doi.org/10.1080/20964471.2019.1625151
    """  # noqa: E501

    filename_glob = "osmwater100m_1_0_NoData.tif"
    filename_regex = r"^.{27}"
    separate_files = False

    url = "ftp://ftp.worldpop.org.uk/GIS/Covariates/Global_2000_2020/0_Mosaicked/OSM_Water/BinaryMosaic_1_0_NoData/osmwater100m_1_0_NoData.tif"  # noqa: E501

    md5s = "4d168bf8dfdae4dc33ca31c0eac50865"
