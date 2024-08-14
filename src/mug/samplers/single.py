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
import torch

from random import choice

from typing import Iterable, Iterator, Optional, Tuple, Union

from torch.utils.data import Sampler

from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import GeoSampler, GridGeoSampler, RandomGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.samplers.utils import get_random_bounding_box, tile_to_chips


class MultiSampler(Sampler[BoundingBox]):
    def __init__(self, samplers: Iterable[GeoSampler]) -> None:
        self.length = sum(map(len, samplers))
        self.samplers = samplers
    
    def __len__(self) -> int:
        return self.length


class RandomMultiSampler(MultiSampler):
    def __iter__(self) -> Iterator[BoundingBox]:
        sampler_set = set(list(iter(sampler) for sampler in self.samplers))
        sampler_list = list(sampler_set)
        while len(sampler_set) > 0:
            sampler = choice(sampler_list)
            try:
                yield next(sampler)
            except StopIteration:
                sampler_set.remove(sampler)
                sampler_list = list(sampler_set)


class SequentialMultiSampler(MultiSampler):
    def __iter__(self) -> Iterator[BoundingBox]:
        sampler_iterables = list(iter(sampler) for sampler in self.samplers)
        for sampler in sampler_iterables:
            for sample in sampler:
                yield sample


class ForecastingRandomGeoSampler(RandomGeoSampler):
    def __iter__(self) -> Iterator[BoundingBox]:
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            random_bbox = get_random_bounding_box(bounds, self.size, self.res)

            forecasting_bbox = BoundingBox(
                random_bbox.minx,
                random_bbox.maxx,
                random_bbox.miny,
                random_bbox.maxy,
                self.roi.mint,
                self.roi.maxt
            )

            yield forecasting_bbox


class ForecastingGridGeoSampler(GridGeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        super().__init__(dataset, size, stride, roi, units)
        
        bbox = BoundingBox(*self.hits[0].bounds)
        for hit in self.hits:
            bbox |= BoundingBox(*hit.bounds)
        
        rows, cols = tile_to_chips(bbox, self.size, self.stride)
        self.length = rows * cols

    def __iter__(self) -> Iterator[BoundingBox]:
        # For each tile...
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]
                if maxy > bounds.maxy:
                    maxy = bounds.maxy
                    miny = bounds.maxy - self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    if maxx > bounds.maxx:
                        maxx = bounds.maxx
                        minx = bounds.maxx - self.size[1]

                    yield BoundingBox(minx, maxx, miny, maxy, self.roi.mint, self.roi.maxt)
