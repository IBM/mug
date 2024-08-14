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
import os
import glob
import re

from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from rasterio import CRS

from torchgeo.datasets import BoundingBox, GeoDataset, IntersectionDataset, RasterDataset
from torchgeo.datasets.utils import disambiguate_timestamp, stack_samples

from .utils import concat_samples


class SpatiotemporalIntersectionDataset(IntersectionDataset):
    def __init__(
        self,
        dataset1: GeoDataset,
        dataset2: GeoDataset,
        collate_fn: Callable[
            [Sequence[Dict[str, Any]]], Dict[str, Any]
        ] = concat_samples,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new SpatiotemporalIntersectionDataset instance.

        Args:
            dataset1: the first dataset
            dataset2: the second dataset
            collate_fn: function used to collate samples
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            ValueError: if either dataset is not a :class:`GeoDataset`

        """
        super().__init__(dataset1, dataset2, collate_fn, transforms)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # All datasets are guaranteed to have a valid query
        # We constrain our query to IntersectionDataset bounds
        samples = [ds[query & self.bounds] for ds in self.datasets]

        sample = self.collate_fn(samples, dim=1)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __and__(self, other):
        return SpatiotemporalIntersectionDataset(self, other, collate_fn=concat_samples)


class SpatiotemporalRasterDataset(RasterDataset):
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(root, crs, res, bands, transforms, cache)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve sequence and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of sequence and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(List[str], [hit.object for hit in hits])
        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        data_list: List[Tensor] = []      # holds stacked tensors
                                          # over the band dimension

        samples: List[Dict] = []          # list of samples to be stacked
                                          # along the time dimension

        timestamps: Dict[float] = dict()  # each key represents a timestamp
                                          # shared by all files in the corresponding list

        band_filepaths: Dict[str] = []    #

        for filepath in filepaths:
            filename = os.path.basename(filepath)
            directory = os.path.dirname(filepath)
            match = re.match(filename_regex, filename)
            if self.separate_files:  # re: bands only
                for band in self.bands:
                    if match:
                        if "band" in match.groupdict():
                            start = match.start("band")
                            end = match.end("band")
                            filename = filename[:start] + band + filename[end:]
                        if "date" in match.groupdict():
                            date = match.group("date")
                            timestamp, _ = disambiguate_timestamp(date, self.date_format)
                            if timestamp not in timestamps:
                                timestamps[timestamp] = dict()
                            if band not in timestamps[timestamp]:
                                timestamps[timestamp][band] = list()

                    filepath = glob.glob(os.path.join(directory, filename))[0]

                    if band not in band_filepaths:
                        band_filepaths[band] = list()
                    band_filepaths[band].append(filepath)

                    if "date" in match.groupdict():
                        timestamps[timestamp][band].append(filepath)
            else:
                if match:
                    if "date" in match.groupdict():
                        date = match.group("date")
                        timestamp, _ = disambiguate_timestamp(date, self.date_format)
                        if timestamp not in timestamps:
                            timestamps[timestamp] = list()
                        timestamps[timestamp].append(filepath)

        if len(timestamps) > 0:
            for timestamp in sorted(timestamps.keys()):
                if self.separate_files:
                    for band in self.bands:
                        data_list.append(self._merge_files(timestamps[timestamp][band], query))
                    data = torch.cat(data_list)
                else:
                    data = self._merge_files(timestamps[timestamp], query, self.band_indexes)
                
                sample = {
                    "crs": self.crs,
                    "bbox": query,
                    "sequence": data.float()
                }
                                
                samples.append(sample)
        else:
            if self.separate_files:
                for band in self.bands:
                    data_list.append(self._merge_files(band_filepaths[band], query))
                data = torch.cat(data_list)
            else:
                data = self._merge_files(filepaths, query, self.band_indexes)
            
            sample = {
                "crs": self.crs,
                "bbox": query,
                "sequence": data.float()
            }
                            
            samples.append(sample)

        return \
            self.transforms(stack_samples(samples)) \
            if self.transforms is not None \
            else stack_samples(samples)

    def __and__(self, other):
        return SpatiotemporalIntersectionDataset(self, other, collate_fn=concat_samples)


class ForecastingDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        variates: List[SpatiotemporalRasterDataset],
        covariates: List[RasterDataset] = None
    ) -> None:
        """Initialize a new ForecastingDataset instance.

        Args:
            variates: list of SpatiotemporalRasterDataset objects
                with sequences to be forecast
            covariates: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)

        Raises:
            TypeError: if any of the objects in ``variates`` are
                not of SpatiotemporalRasterDataset type 
        """
        self.variates_metadata = []
        self.covariates_metadata = []

        for ds in variates:
            self.variates_metadata.append(str(type(ds)))
            # TODO: Store metadata for each variate

        for ds in covariates:
            self.covariates_metadata.append(str(type(ds)))
            # TODO: Store metadata for each covariate

        dynamic_covariates = [ds for ds in covariates if isinstance(ds, SpatiotemporalRasterDataset)]
        static_covariates = [ds for ds in covariates if not isinstance(ds, SpatiotemporalRasterDataset)]

        # Union[SpatiotemporalRasterDataset, SpatiotemporalIntersectionDataset]
        self.variates = variates.pop(0)
        for ds in variates: self.variates &= ds

        # Union[SpatiotemporalRasterDataset, SpatiotemporalIntersectionDataset, None]
        self.dynamic_covariates = dynamic_covariates.pop(0) if len(dynamic_covariates) > 0 else None
        for ds in dynamic_covariates: self.dynamic_covariates &= ds

        # Union[RasterDataset, IntersectionDataset, None]
        self.static_covariates = static_covariates.pop(0) if len(static_covariates) > 0 else None
        for ds in static_covariates: self.static_covariates &= ds

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve variate/covariate and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of variate/covariate and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        sample = self.variates[query]
        sample["variates"] = sample.pop("sequence")

        if self.dynamic_covariates is not None:
            sample_dynamic = self.dynamic_covariates[query]
            sample["covariates"] = sample_dynamic["sequence"]
        if self.static_covariates is not None:
            sample_static = self.static_covariates[query]
            k = sample["variates"].shape[0]
            sample_static["image"] = sample_static["image"].unsqueeze(0).repeat(k, 1, 1, 1)
            if "covariates" not in sample.keys():
                sample["covariates"] = sample_static["image"]
            else:
                sample["covariates"] = torch.cat((sample["covariates"], sample_static["image"]), dim=1)
        
        return sample

    def __and__(self, other: Dataset):
        raise NotImplementedError("Operator not supported.")

    def __or__(self, other: Dataset):
        raise NotImplementedError("Operator not supported.")

    def __len__(self):
        return len(self.variates)

    def __str__(self):
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: Dataset
    bbox: {self.variates.bounds}
    size: {len(self.variates)}
    target: {self.variates.__class__.__name__}"""