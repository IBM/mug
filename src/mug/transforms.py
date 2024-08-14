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
from typing import Any, Dict, Iterable


def factor(sample: Dict[Any, Any], weight: float) -> Dict[Any, Any]:
    """Factor data by a given weight.
    """
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value * weight
    
    return sample

def normalize_binary_land_mask(sample: Dict[Any, Any]) -> Dict[Any, Any]:
    """Normalize land-nodata masks as binary masks.
    """
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            value = torch.where(value == 0., 1./2., value)
            value = torch.where(value == 255., 0., value)
            sample[key] = torch.clone(value)
    
    return sample

def reclassify_time_differences(sample: Dict[Any, Any]) -> Dict[Any, Any]:
    """Reclassify pixels that differ between two consecutive time steps.
    """
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if not torch.max(value) > 1:
                for j in reversed(range(1, value.shape[0])):
                    dt = - (value[j-1,:] - value[j,:]) * 1./2.
                    sample[key][j,:] = torch.clone(value[j,:] - dt)
            else:
                raise ValueError("Tensor not normalized.")
    
    return sample

def chain(sample: Dict[Any, Any], transforms: Iterable) -> Dict[Any, Any]:
    """Execute chain of transforms.
    """
    for transform in transforms:
        sample = transform(sample)
    
    return sample