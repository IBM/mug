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
import importlib
import inspect

@click.group()
@click.pass_context
@click.option("-b", "--batch-size", type=int, default=32, help="Batch size.")
@click.option("datasets", "-d", "--dataset", type=str, multiple=True, help="Explanatory dataset.")
@click.option("-e", "--epochs", type=int, default=16, help="Maximum number of epochs.")
@click.option("-r", "--learning-rate", type=float, default=1e-3, help="Learning rate.")
@click.option("-l", "--length", type=int, default=5, help="Sequence length.")
@click.option("-t", "--training-samples", type=int, default=512, show_default=True, help="Number of training samples.")
@click.option("-v", "--validation-samples", type=int, default=128, show_default=True, help="Number of validation samples.")
def fit(
    ctx,
    batch_size: int,
    datasets: str,
    epochs: int,
    learning_rate: float,
    length: int,
    training_samples: int,
    validation_samples: int,
    ) -> None:
    """Model training."""
    ctx.obj['BATCH_SIZE'] = batch_size
    ctx.obj['DATASETS'] = datasets
    ctx.obj['EPOCHS'] = epochs
    ctx.obj['LEARNING_RATE'] = learning_rate
    ctx.obj['SEQUENCE'] = length
    ctx.obj['TRAINING_SAMPLES'] = training_samples
    ctx.obj['VALIDATION_SAMPLES'] = validation_samples


models = importlib.import_module(".models", package="mug")
for m in inspect.getmembers(models):
    if inspect.ismodule(m[1]):
        try:
            fit.add_command(getattr(getattr(models, m[0]), "fit"))
        except:
            pass