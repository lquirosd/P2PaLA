## How to use
For inference using a pre-trained model follow this steps:
1. Put all your images under a folder (symlinks are allowed). For example ```data```

2. Download the pre-trained model (see available model bellow), and it's respective config file.

3. Run P2PaLA:
```bash
python P2PaLA.py --config <path_to_config_file> --prev_model <path_to_model> --prod_data <pointer_to_your_images>
```

> Note: This command will force to use GPU, if you want to use CPU just add ```--gpu -1``` 

## Available models

### Baselines only
* ALAR:

    - model: 

    ```wget --no-check-certificate  https://www.prhlt.upv.es/~lquirosd/P2PaLA/ALAR_min_model_17_12_18.pth```
    - config: 

    ```wget --no-check-certificate  https://www.prhlt.upv.es/~lquirosd/P2PaLA/config_ALAR_min_model_17_12_18_inference.txt```

### Baselines and Zones
WIP

### Zones only
WIP

### Tables
WIP
