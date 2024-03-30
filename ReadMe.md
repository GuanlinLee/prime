# Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- pillow
- diffusers
- ftfy
- transformers
- opencv-python
- numpy
- pyyaml
- accelerate
- xformers
- av
- kornia

# Usage

First download the Validation set of Imagenet. 
Put it into the folder `./val`.

We provide a template config file in the folder. The config file contains all the parameters needed to run the code.
You can modify the config file to change the path to the dataset, the path of your videos.
```
python PRIME.py --config_path <path to config file>
``` 

# Videos used in our experiments

We cannot release the video dataset, due to the legal restrictions of the original video owners.
However, we provide discription files for the videos used in our experiments in the folder `./configs`.
These config files discribe the video content and the prompt we used to edit them.
If you do need the original videos, please contact us by email.

# Legal issues

Please do not redistribute any videos.
This code is only for research purposes.
