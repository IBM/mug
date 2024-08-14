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
import sys
import click
import importlib
import inspect


@click.command()
@click.pass_context
def datasets(ctx) -> None:
    """Shows available explanatory datasets."""

    module = importlib.import_module(".datasets.worldpop", package="mug")
    staticds = getattr(module, "_StaticDataset")
    dynds = getattr(module, "_DynamicDataset")
    for m in inspect.getmembers(module):
        if inspect.isclass(m[1]):
            if issubclass(m[1], (staticds, dynds)):
                if m[0] not in ["_StaticDataset", "_DynamicDataset", "UrbanLandCover"]:
                    print(m[0])
