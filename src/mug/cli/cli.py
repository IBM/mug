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
import click
import random
import numpy as np
import torch

from .datasets.commands import datasets
from .fit.commands import fit
from .predict.commands import predict
from .roi.commands import roi

@click.group()
@click.pass_context
@click.option("--checkpoint", type=click.Path(exists=True), help="Checkpoint file.")
@click.option("countries", "-c", "--country", type=str, multiple=True, help="Two-letter country code.")
@click.option("--data", type=click.Path(exists=True), default="data", show_default=True, help="Directory containing data files.")
@click.option("--debug", is_flag=True, help="Debug mode.")
@click.option("--output", type=click.Path(), default="output", show_default=True, help="Output directory.")
@click.option("--roi", type=click.Path(exists=True), default="rois.csv", help="Path to ROI file.")
@click.option("--samples", type=int, default=32, show_default=True, help="Number of samples per ROI.")
@click.option("--sampling", type=click.Choice(("grid","random"), case_sensitive=False), default="random", show_default=True, help="Spatial sampling strategy.")
@click.option("--seed", type=int, default=None, help="Random number generator seed.")
@click.option("--year", type=int, default=2000, show_default=True, help="If fitting, this is the first year of the sequence. If predicting, this is the target year.")
def main(
    ctx,
    checkpoint: str,
    countries: str,
    data: str,
    debug: bool,
    output: str,
    roi: str,
    samples: int,
    sampling: str,
    seed: int,
    year: int,
    ) -> None:
    """Models of Urban Growth"""
    ctx.ensure_object(dict)

    ctx.obj['CHECKPOINT_FILE'] = checkpoint
    ctx.obj['DATA_PATH'] = data
    ctx.obj['DEBUG'] = debug
    ctx.obj['OUTPUT_PATH'] = output
    ctx.obj['SAMPLING'] = sampling
    ctx.obj['YEAR'] = year
    ctx.obj['ROI_FILE'] = roi
    ctx.obj['ROI_COUNTRIES'] = countries
    ctx.obj['ROI_SAMPLES'] = samples

    # reproducibility tricks
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

main.add_command(datasets)
main.add_command(fit)
main.add_command(predict)
main.add_command(roi)