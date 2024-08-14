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
from torch import Tensor
from typing import Any, Dict, Iterable, Tuple
from torchgeo.datasets.utils import _list_dict_to_dict_list, stack_samples


def concat_samples(samples: Iterable[Dict[Any, Any]], dim: int = 0) -> Dict[Any, Any]:
    """Concatenate a list of samples along a given axis :param:`dim`.

    Useful for joining samples in a :class:`torchgeo.datasets.IntersectionDataset`.

    Args:
        samples: list of samples
        dim: dimension to be concatenated

    Returns:
        a single sample
    """
    collated: Dict[Any, Any] = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            collated[key] = torch.cat(value, dim=dim)
        else:
            collated[key] = value[0]
    return collated

def unbind_sequence(
    samples: Iterable[Dict[Any, Any]],
    target_sequences: int = 1
) -> Tuple[Tensor, Tensor]:
    """Unbind time dimension of a list of samples into two sequences.

    Useful for forming a mini-batch of samples to pass to
    :class:`torch.utils.data.DataLoader`.

    Args:
        samples: list of samples
        target_sequences: number of target sequences

    Returns:
        a tuple of batch and target tensors
    """
    sample = stack_samples(samples)

    # check if ForecastingDataset sample
    if "variates" not in sample.keys():
        raise TypeError("Not a ForecastingDataset sample")

    batch = sample["variates"][:,:-target_sequences,:,:,:]
    target = sample["variates"][:,-target_sequences:,:,:,:]

    if "covariates" in sample.keys():
        batch = torch.cat((batch, sample["covariates"][:,:-target_sequences,:,:,:]), dim=2)
        target = torch.cat((target, sample["covariates"][:,-target_sequences:,:,:,:]), dim=2)

    return batch, target