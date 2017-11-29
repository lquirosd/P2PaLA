P2PaLA :scroll:
======

Page to PAGE Layout Analysis Tool (P2PaLA) is a tool for Document Layout Analysis based in Neural Networks.

Description
===========

Requirements
===========
    * Linux
    * PyTorch
    * TensorBoard
    * OpenCV

Usage
=====
```text
usage: P2PaLA.py [-h] [--config CONFIG] [--exp_name EXP_NAME]
                 [--work_dir WORK_DIR]
                 [--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                 [--baseline_evaluator BASELINE_EVALUATOR]
                 [--num_workers NUM_WORKERS] [--gpu GPU] [--no_display]
                 [--use_global_log USE_GLOBAL_LOG] [--log_comment LOG_COMMENT]
                 [--img_size IMG_SIZE IMG_SIZE] [--line_color LINE_COLOR]
                 [--line_width LINE_WIDTH] [--regions REGIONS [REGIONS ...]]
                 [--merge_regions MERGE_REGIONS [MERGE_REGIONS ...]]
                 [--batch_size BATCH_SIZE]
                 [--shuffle_data | --no-shuffle_data]
                 [--pin_memory | --no-pin_memory] [--flip_img | --no-flip_img]
                 [--input_channels INPUT_CHANNELS]
                 [--output_channels OUTPUT_CHANNELS] [--cnn_ngf CNN_NGF]
                 [--use_gan | --no-use_gan] [--gan_layers GAN_LAYERS]
                 [--loss_lambda LOSS_LAMBDA] [--g_loss {L1,MSE,smoothL1}]
                 [--adam_lr ADAM_LR] [--adam_beta1 ADAM_BETA1]
                 [--adam_beta2 ADAM_BETA2] [--do_train | --no-do_train]
                 [--cont_train] [--prev_model PREV_MODEL] [--tr_data TR_DATA]
                 [--epochs EPOCHS] [--tr_img_list TR_IMG_LIST]
                 [--tr_label_list TR_LABEL_LIST] [--do_test | --no-do_test]
                 [--te_data TE_DATA] [--te_img_list TE_IMG_LIST]
                 [--te_label_list TE_LABEL_LIST] [--do_val | --no-do_val]
                 [--val_data VAL_DATA] [--val_img_list VAL_IMG_LIST]
                 [--val_label_list VAL_LABEL_LIST] [--do_prod | --no-do_prod]
                 [--prod_data PROD_DATA] [--prod_img_list PROD_IMG_LIST]

NN Implentation for Layout Analysis

optional arguments:
  -h, --help            show this help message and exit

General Parameters:
  --config CONFIG       Use this configuration file (default: None)
  --exp_name EXP_NAME   Name of the experiment. Models and data will be stored
                        into a folder under this name (default: layout_exp)
  --work_dir WORK_DIR   Where to place output data (default: ./work/)
  --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: INFO)
  --baseline_evaluator BASELINE_EVALUATOR
                        Command to evaluate baselines (default: ['java',
                        '-jar', '/home/lquirosd/REPOS/TranskribusBaseLineEvalu
                        ationScheme/TranskribusBaseLineEvaluationScheme_v0.1.0
                        /TranskribusBaseLineEvaluationScheme-0.1.0-jar-with-
                        dependencies.jar', '-no_s'])
  --num_workers NUM_WORKERS
                        Number of workers used to proces input data. If not
                        provided all available CPUs will be used. (default: 4)
  --gpu GPU             GPU id. Use -1 to disable. Only 1 GPU setup is
                        available for now ;( (default: 0)
  --no_display          Do not display data on TensorBoard (default: False)
  --use_global_log USE_GLOBAL_LOG
                        Save TensorBoard log on this folder instead default
                        (default: )
  --log_comment LOG_COMMENT
                        Add this commaent to TensorBoard logs name (default: )

Data Related Parameters:
  --img_size IMG_SIZE IMG_SIZE
                        Scale images to this size. Format --img_size H W
                        (default: [1024, 768])
  --line_color LINE_COLOR
                        Draw GT lines using this color, range [1,254]
                        (default: 128)
  --line_width LINE_WIDTH
                        Draw GT lines using this number of pixels (default:
                        10)
  --regions REGIONS [REGIONS ...]
                        List of regions to be extracted. Format: --regions r1
                        r2 r3 ... (default: ['$tip', '$par', '$not', '$nop',
                        '$pag'])
  --merge_regions MERGE_REGIONS [MERGE_REGIONS ...]
                        Merge regions on PAGE file into a single one. Format
                        --merge_regions r1:r2,r3 r4:r5, then r2 and r3 will be
                        merged into r1 and r5 into r4 (default: None)

Data Loader Parameters:
  --batch_size BATCH_SIZE
                        Number of images per mini-batch (default: 6)
  --shuffle_data        Suffle data during training (default: True)
  --no-shuffle_data     Do not suffle data during training (default: True)
  --pin_memory          Pin memory before send to GPU (default: True)
  --no-pin_memory       Pin memory before send to GPU (default: True)
  --flip_img            Randomly flip images during training (default: False)
  --no-flip_img         Do not randomly flip images during training (default:
                        False)

Neural Networks Parameters:
  --input_channels INPUT_CHANNELS
                        Number of channels of input data (default: 3)
  --output_channels OUTPUT_CHANNELS
                        Number of channels of labels (default: 2)
  --cnn_ngf CNN_NGF     Number of filters of CNNs (default: 64)
  --use_gan             USE GAN to compute G loss (default: True)
  --no-use_gan          do not use GAN to compute G loss (default: True)
  --gan_layers GAN_LAYERS
                        Number of layers of GAN NN (default: 3)
  --loss_lambda LOSS_LAMBDA
                        Lambda weith to copensate GAN vs G loss (default:
                        0.001)
  --g_loss {L1,MSE,smoothL1}
                        Loss function for G NN (default: L1)

Optimizer Parameters:
  --adam_lr ADAM_LR     Initial Lerning rate for ADAM opt (default: 0.001)
  --adam_beta1 ADAM_BETA1
                        First ADAM exponential decay rate (default: 0.5)
  --adam_beta2 ADAM_BETA2
                        Secod ADAM exponential decay rate (default: 0.999)

Training Parameters:
  --do_train            Run train stage (default: True)
  --no-do_train         Do not run train stage (default: True)
  --cont_train          Continue training using this model (default: False)
  --prev_model PREV_MODEL
                        Use this previously trainned model (default: None)
  --tr_data TR_DATA     Train data folder. Train images are expected there,
                        also PAGE XML files are expected to be in
                        --tr_data/page folder (default: ./data/train/)
  --epochs EPOCHS       Number of training epochs (default: 100)
  --tr_img_list TR_IMG_LIST
                        List to all images ready to be used by NN train, if
                        not provide it will be generated from original data.
                        (default: )
  --tr_label_list TR_LABEL_LIST
                        List to all label ready to be used by NN train, if not
                        provide it will be generated from original data.
                        (default: )

Test Parameters:
  --do_test             Run test stage (default: False)
  --no-do_test          Do not run test stage (default: False)
  --te_data TE_DATA     Test data folder. Test images are expected there, also
                        PAGE XML files are expected to be in --te_data/page
                        folder (default: ./data/test/)
  --te_img_list TE_IMG_LIST
                        List to all images ready to be used by NN train, if
                        not provide it will be generated from original data.
                        (default: )
  --te_label_list TE_LABEL_LIST
                        List to all label ready to be used by NN train, if not
                        provide it will be generated from original data.
                        (default: )

Validation Parameters:
  --do_val              Run Validation stage (default: False)
  --no-do_val           do not run Validation stage (default: False)
  --val_data VAL_DATA   Validation data folder. Validation images are expected
                        there, also PAGE XML files are expected to be in
                        --te_data/page folder (default: ./data/val/)
  --val_img_list VAL_IMG_LIST
                        List to all images ready to be used by NN train, if
                        not provide it will be generated from original data.
                        (default: )
  --val_label_list VAL_LABEL_LIST
                        List to all label ready to be used by NN train, if not
                        provide it will be generated from original data.
                        (default: )

Production Parameters:
  --do_prod             Run production stage (default: False)
  --no-do_prod          Do not run production stage (default: False)
  --prod_data PROD_DATA
                        Production data folder. Production images are expected
                        there. (default: ./data/prod/)
  --prod_img_list PROD_IMG_LIST
                        List to all images ready to be used by NN train, if
                        not provide it will be generated from original data.
                        (default: )
```
Authors
=======


License
=======

GNU General Public License v3.0
See [LICENSE](LICENSE) to see the full text.

Acknowledgments
===============

