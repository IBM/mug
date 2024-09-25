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
def predict(ctx):
    """Model inferencing."""
    print(ctx.obj)

#models = importlib.import_module(".models", package="mug")
models = importlib.import_module(".nn", package="mug")
for m in inspect.getmembers(models):
    if inspect.ismodule(m[1]):
        try:
            predict.add_command(getattr(getattr(models, m[0]), "predict"))
        except:
            pass