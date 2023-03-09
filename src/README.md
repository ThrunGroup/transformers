<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Evaluation Suite

### Updates

**March 9, 2023**

🤗 New Features:

1. 🚀 **The code now allows you to specify which layers to accelerate.** By providing a comma-separated list of layer 
   indices 
or index ranges, the code will automatically parse it and apply the accelerator to specific layers in apply_accelerator (link). For instance, if you specify "0-7, 11", the code will only accelerate the fully connected layers in 0~7th and 11th block.
2. 🥶 Similarly, **you can specify which layers to freeze** by providing a comma-separated list of layer indices or 
   index 
   ranges, which will be parsed automatically, and freeze the specified layers in freeze_layers (link).
3. 🏋️‍♀️ **A new flag train_accelerated_layers has been added.** By default, it is set to False, meaning that the 
   layers where 
   the accelerator was applied will be frozen during training. However, if you set this flag to True, the frozen layers will be trained as well.
4. 🤖 The tool now includes an **automatic naming and saving feature for the model**, making it easier to load the model 
   later. The model is named based on its configuration, including which layers were accelerated or frozen, and 
   whether or not the accelerated layers were trained. For example, a model may be named "gpt2_SVD_accelerated_11_froze_0-8_trained_accelerated_layers".



