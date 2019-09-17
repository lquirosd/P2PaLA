```bash
usage: P2PaLA.py [-h] [--config CONFIG] [--exp_name EXP_NAME]
                 [--work_dir WORK_DIR]
                 [--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                 [--num_workers NUM_WORKERS] [--gpu GPU] [--seed SEED]
                 [--no_display] [--use_global_log USE_GLOBAL_LOG]
                 [--log_comment LOG_COMMENT] [--img_size IMG_SIZE IMG_SIZE]
                 [--line_color LINE_COLOR] [--line_width LINE_WIDTH]
                 [--regions REGIONS [REGIONS ...]]
                 [--merge_regions MERGE_REGIONS [MERGE_REGIONS ...]]
                 [--nontext_regions NONTEXT_REGIONS [NONTEXT_REGIONS ...]]
                 [--region_type REGION_TYPE [REGION_TYPE ...]]
                 [--approx_alg {optimal,trace}] [--num_segments NUM_SEGMENTS]
                 [--max_vertex MAX_VERTEX] [--line_offset LINE_OFFSET]
                 [--min_area MIN_AREA] [--save_prob_mat SAVE_PROB_MAT]
                 [--line_alg {basic,external}] [--batch_size BATCH_SIZE]
                 [--shuffle_data | --no-shuffle_data]
                 [--pin_memory | --no-pin_memory] [--flip_img | --no-flip_img]
                 [--elastic_def ELASTIC_DEF] [--e_alpha E_ALPHA]
                 [--e_stdv E_STDV] [--affine_trans AFFINE_TRANS]
                 [--t_stdv T_STDV] [--r_kappa R_KAPPA] [--sc_stdv SC_STDV]
                 [--sh_kappa SH_KAPPA] [--trans_prob TRANS_PROB]
                 [--do_prior DO_PRIOR] [--input_channels INPUT_CHANNELS]
                 [--out_mode {L,R,LR}] [--cnn_ngf CNN_NGF]
                 [--use_gan | --no-use_gan] [--gan_layers GAN_LAYERS]
                 [--loss_lambda LOSS_LAMBDA] [--g_loss {L1,MSE,smoothL1}]
                 [--net_out_type {C,R}] [--adam_lr ADAM_LR]
                 [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                 [--do_train | --no-do_train] [--cont_train]
                 [--prev_model PREV_MODEL] [--save_rate SAVE_RATE]
                 [--tr_data TR_DATA] [--epochs EPOCHS]
                 [--tr_img_list TR_IMG_LIST] [--tr_label_list TR_LABEL_LIST]
                 [--fix_class_imbalance FIX_CLASS_IMBALANCE]
                 [--weight_const WEIGHT_CONST] [--do_test | --no-do_test]
                 [--te_data TE_DATA] [--te_img_list TE_IMG_LIST]
                 [--te_label_list TE_LABEL_LIST] [--do_off DO_OFF]
                 [--do_val | --no-do_val] [--val_data VAL_DATA]
                 [--val_img_list VAL_IMG_LIST]
                 [--val_label_list VAL_LABEL_LIST] [--do_prod | --no-do_prod]
                 [--prod_data PROD_DATA] [--prod_img_list PROD_IMG_LIST]
                 [--target_list TARGET_LIST] [--hyp_list HYP_LIST]

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
  --num_workers NUM_WORKERS
                        Number of workers used to proces input data. If not
                        provided all available CPUs will be used. (default: 4)
  --gpu GPU             GPU id. Use -1 to disable. Only 1 GPU setup is
                        available for now ;( (default: 0)
  --seed SEED           Set manual seed for generating random numbers
                        (default: 5)
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
                        merged into r1 and r5 into r4 (default: {})
  --nontext_regions NONTEXT_REGIONS [NONTEXT_REGIONS ...]
                        List of regions where no text lines are expected.
                        Format: --nontext_regions r1 r2 r3 ... (default: None)
  --region_type REGION_TYPE [REGION_TYPE ...]
                        Type of region on PAGE file. Format --region_type
                        t1:r1,r3 t2:r5, then type t1 will assigned to regions
                        r1 and r3 and type t2 to r5 and so on... (default:
                        None)
  --approx_alg {optimal,trace}
                        Algorith to approximate baseline to N segments.
                        optimal: [Perez & Vidal, 1994] algorithm. trace: Use
                        trace normalization algorithm. (default: optimal)
  --num_segments NUM_SEGMENTS
                        Number of segments of the output baseline (default: 4)
  --max_vertex MAX_VERTEX
                        Maximun number of vertex used to approximate the
                        baselined when use 'optimal' algorithm (default: 10)
  --line_offset LINE_OFFSET
                        Fixed width of polygon around each baseline. (default:
                        50)
  --min_area MIN_AREA   Minimum allowed area for Zone extraction, as a
                        percentage of <--image_size> (default: 0.01)
  --save_prob_mat SAVE_PROB_MAT
                        Save Network Prob Matrix at Inference (default: False)
  --line_alg {basic,external}
                        Algorithm used during baseline detection, Stage 2
                        (default: basic)

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
  --elastic_def ELASTIC_DEF
                        Use elastic deformation during training (default:
                        True)
  --e_alpha E_ALPHA     alpha value for elastic deformations (default: 0.045)
  --e_stdv E_STDV       std dev value for elastic deformations (default: 4)
  --affine_trans AFFINE_TRANS
                        Use affine transformations during training (default:
                        True)
  --t_stdv T_STDV       std deviation of normal dist. used in translate
                        (default: 0.02)
  --r_kappa R_KAPPA     concentration of von mises dist. used in rotate
                        (default: 30)
  --sc_stdv SC_STDV     std deviation of log-normal dist. used in scale
                        (default: 0.12)
  --sh_kappa SH_KAPPA   concentration of von mises dist. used in shear
                        (default: 20)
  --trans_prob TRANS_PROB
                        probabiliti to perform a transformation (default: 0.5)
  --do_prior DO_PRIOR   Compute prior distribution over classes (default:
                        False)

Neural Networks Parameters:
  --input_channels INPUT_CHANNELS
                        Number of channels of input data (default: 3)
  --out_mode {L,R,LR}   Type of output: L: Only Text Lines will be extracted
                        R: Only Regions will be extracted LR: Lines and
                        Regions will be extracted (default: LR)
  --cnn_ngf CNN_NGF     Number of filters of CNNs (default: 64)
  --use_gan             USE GAN to compute G loss (default: True)
  --no-use_gan          do not use GAN to compute G loss (default: True)
  --gan_layers GAN_LAYERS
                        Number of layers of GAN NN (default: 3)
  --loss_lambda LOSS_LAMBDA
                        Lambda weith to copensate GAN vs G loss (default: 100)
  --g_loss {L1,MSE,smoothL1}
                        Loss function for G NN (default: L1)
  --net_out_type {C,R}  Compute the problem as a classification or Regresion:
                        C: Classification R: Regresion (default: C)

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
  --save_rate SAVE_RATE
                        Save checkpoint each --save_rate epochs (default: 10)
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
  --fix_class_imbalance FIX_CLASS_IMBALANCE
                        use weights at loss function to handle class
                        imbalance. (default: True)
  --weight_const WEIGHT_CONST
                        weight constant to fix class imbalance (default: 1.02)

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
  --do_off DO_OFF       Turn DropOut Off during inference (default: True)

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

Evaluation Parameters:
  --target_list TARGET_LIST
                        List of ground-truth PAGE-XML files (default: )
  --hyp_list HYP_LIST   List of hypotesis PAGE-XMLfiles (default: )
```
