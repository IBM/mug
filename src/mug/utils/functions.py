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
from typing import List


def pred_mask(input, batch_size, size):
    expected_shape = (batch_size, size, size)

    # Checking the dimensions
    if input.shape != expected_shape:
        raise ValueError(f"Input tensor shape {input.shape} does not match the expected shape {expected_shape}")
    
    # Create a mask based on the output of the model
    mask_1 = (input >= 0.0) & (input < 0.15)  # 0 water
    mask_2 = (input >= 0.15) & (input < 0.60) # 1 land
    mask_3 = (input >= 0.60) & (input < 0.80) # 2 non-bs to bs
    mask_4 = (input >= 0.80) & (input < 1.0)  # 3 bs

    # Initialize an output tensor with zeros
    mapped_input = torch.zeros_like(input).to(torch.int) # convert to integer

    # Apply the mapping conditions for y_hat

    mapped_input[mask_1] = 0 # 0.0 water
    mapped_input[mask_2] = 1 # 0.5 land
    mapped_input[mask_3] = 2 # 0.75 no bs to bs
    mapped_input[mask_4] = 3 # 1.0 bs

    return mapped_input


def target_mask(input, batch_size, size):
    
    expected_shape = (batch_size, size, size)

    # Checking the dimensions
    if input.shape != expected_shape:
        raise ValueError(f"Input tensor shape {input.shape} does not match the expected shape {expected_shape}")

    mask_1 = (input == 0.0)
    mask_2 = (input == 0.5)
    mask_3 = (input == 0.75)
    mask_4 = (input == 1.0)
    
    mapped_input = torch.zeros_like(input).to(torch.int)

    mapped_input[mask_1] = 0 # 0.0 water
    mapped_input[mask_2] = 1 # 0.5 land
    mapped_input[mask_3] = 2 # 0.75 no bs to bs
    mapped_input[mask_4] = 3 # 1.0 bs
    

    return mapped_input

def class_mask(n_classes, y):
    
    # Initialize the mask tensor
    mask = torch.zeros((n_classes,) + y.shape)
    
    # Fill the mask tensor for each class
    for class_indx in range(n_classes):
        mask[class_indx] = (y == class_indx)
        
    return mask