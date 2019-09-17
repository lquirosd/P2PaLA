from __future__ import print_function
from __future__ import division

import numpy as np
from collections import OrderedDict
import argparse
import os

# from math import log
import multiprocessing
import logging

from . import art
from evalTools.metrics import levenshtein


class Arguments(object):
    """
    """

    def __init__(self, logger=None):
        """
        """
        self.logger = logger or logging.getLogger(__name__)
        parser_description = """
        NN Implentation for Layout Analysis
        """
        regions = ["$tip", "$par", "$not", "$nop", "$pag"]
        merge_regions = {}
        n_cpus = multiprocessing.cpu_count()

        self.parser = argparse.ArgumentParser(
            description=parser_description,
            fromfile_prefix_chars="@",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.parser.convert_arg_line_to_args = self._convert_file_to_args
        # ----------------------------------------------------------------------
        # ----- Define general parameters
        # ----------------------------------------------------------------------
        general = self.parser.add_argument_group("General Parameters")
        general.add_argument(
            "--config", default=None, type=str, help="Use this configuration file"
        )
        general.add_argument(
            "--exp_name",
            default="layout_exp",
            type=str,
            help="""Name of the experiment. Models and data 
                                       will be stored into a folder under this name""",
        )
        general.add_argument(
            "--work_dir", default="./work/", type=str, help="Where to place output data"
        )
        # --- Removed, input data should be handled by {tr,val,te,prod}_data variables
        # general.add_argument('--data_path', default='./data/',
        #                     type=self._check_in_dir,
        #                     help='path to input data')
        general.add_argument(
            "--log_level",
            default="INFO",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level",
        )
        # general.add_argument('--baseline_evaluator', default=baseline_evaluator_cmd,
        #                     type=str, help='Command to evaluate baselines')
        general.add_argument(
            "--num_workers",
            default=n_cpus,
            type=int,
            help="""Number of workers used to proces 
                                  input data. If not provided all available
                                  CPUs will be used.
                                  """,
        )
        general.add_argument(
            "--gpu",
            default=0,
            type=int,
            help=(
                "GPU id. Use -1 to disable. " "Only 1 GPU setup is available for now ;("
            ),
        )
        general.add_argument(
            "--seed",
            default=5,
            type=int,
            help="Set manual seed for generating random numbers",
        )
        general.add_argument(
            "--no_display",
            default=False,
            action="store_true",
            help="Do not display data on TensorBoard",
        )
        general.add_argument(
            "--use_global_log",
            default="",
            type=str,
            help="Save TensorBoard log on this folder instead default",
        )
        general.add_argument(
            "--log_comment",
            default="",
            type=str,
            help="Add this commaent to TensorBoard logs name",
        )
        # ----------------------------------------------------------------------
        # ----- Define processing data parameters
        # ----------------------------------------------------------------------
        data = self.parser.add_argument_group("Data Related Parameters")
        data.add_argument(
            "--img_size",
            default=[1024, 768],
            nargs=2,
            type=self._check_to_int_array,
            help="Scale images to this size. Format --img_size H W",
        )
        data.add_argument(
            "--line_color",
            default=128,
            type=int,
            help="Draw GT lines using this color, range [1,254]",
        )
        data.add_argument(
            "--line_width",
            default=10,
            type=int,
            help="Draw GT lines using this number of pixels",
        )
        data.add_argument(
            "--regions",
            default=regions,
            nargs="+",
            type=str,
            help="""List of regions to be extracted. 
                               Format: --regions r1 r2 r3 ...""",
        )
        data.add_argument(
            "--merge_regions",
            default=merge_regions,
            nargs="+",
            type=str,
            help="""Merge regions on PAGE file into a single one.
                               Format --merge_regions r1:r2,r3 r4:r5, then r2 and r3
                               will be merged into r1 and r5 into r4""",
        )
        data.add_argument(
            "--nontext_regions",
            default=None,
            nargs="+",
            type=str,
            help="""List of regions where no text lines are expected. 
                               Format: --nontext_regions r1 r2 r3 ...""",
        )
        data.add_argument(
            "--region_type",
            default=None,
            nargs="+",
            type=str,
            help="""Type of region on PAGE file.
                               Format --region_type t1:r1,r3 t2:r5, then type t1
                               will assigned to regions r1 and r3 and type t2 to
                               r5 and so on...""",
        )
        data.add_argument(
            "--approx_alg",
            default="optimal",
            type=str,
            choices=["optimal", "trace"],
            help="""Algorith to approximate baseline to N segments.
                                  optimal: [Perez & Vidal, 1994] algorithm.
                                  trace: Use trace normalization algorithm.""",
        )
        data.add_argument(
            "--num_segments",
            default=4,
            type=int,
            help="Number of segments of the output baseline",
        )
        data.add_argument(
            "--max_vertex",
            default=10,
            type=int,
            help="""Maximun number of vertex used to approximate
                               the baselined when use 'optimal' algorithm""",
        )
        data.add_argument(
            "--line_offset",
            default=50,
            type=int,
            help="""Fixed width of polygon around each baseline.""",
        )
        data.add_argument(
            "--min_area",
            default=0.01,
            type=float,
            help="""Minimum allowed area for Zone extraction,
                    as a percentage of <--image_size>"""
        )
        data.add_argument(
            "--save_prob_mat",
            default=False,
            type=bool,
            help="Save Network Prob Matrix at Inference",
        )
        data.add_argument(
            "--line_alg",
            default='basic',
            type=str,
            choices=["basic","external"],
            help="Algorithm used during baseline detection, Stage 2",
        )
        # ----------------------------------------------------------------------
        # ----- Define dataloader parameters
        # ----------------------------------------------------------------------
        loader = self.parser.add_argument_group("Data Loader Parameters")
        loader.add_argument(
            "--batch_size", default=6, type=int, help="Number of images per mini-batch"
        )
        l_meg1 = loader.add_mutually_exclusive_group(required=False)
        l_meg1.add_argument(
            "--shuffle_data",
            dest="shuffle_data",
            action="store_true",
            help="Suffle data during training",
        )
        l_meg1.add_argument(
            "--no-shuffle_data",
            dest="shuffle_data",
            action="store_false",
            help="Do not suffle data during training",
        )
        l_meg1.set_defaults(shuffle_data=True)
        l_meg2 = loader.add_mutually_exclusive_group(required=False)
        l_meg2.add_argument(
            "--pin_memory",
            dest="pin_memory",
            action="store_true",
            help="Pin memory before send to GPU",
        )
        l_meg2.add_argument(
            "--no-pin_memory",
            dest="pin_memory",
            action="store_false",
            help="Pin memory before send to GPU",
        )
        l_meg2.set_defaults(pin_memory=True)
        l_meg3 = loader.add_mutually_exclusive_group(required=False)
        l_meg3.add_argument(
            "--flip_img",
            dest="flip_img",
            action="store_true",
            help="Randomly flip images during training",
        )
        l_meg3.add_argument(
            "--no-flip_img",
            dest="flip_img",
            action="store_false",
            help="Do not randomly flip images during training",
        )
        l_meg3.set_defaults(flip_img=False)
        loader.add_argument(
            "--elastic_def",
            default=True,
            type=bool,
            help="Use elastic deformation during training",
        )
        loader.add_argument(
            "--e_alpha",
            default=0.045,
            type=float,
            help="alpha value for elastic deformations",
        )
        loader.add_argument(
            "--e_stdv",
            default=4,
            type=float,
            help="std dev value for elastic deformations",
        )
        loader.add_argument(
            "--affine_trans",
            default=True,
            type=bool,
            help="Use affine transformations during training",
        )
        loader.add_argument(
            "--t_stdv",
            default=0.02,
            type=float,
            help="std deviation of normal dist. used in translate",
        )
        loader.add_argument(
            "--r_kappa",
            default=30,
            type=float,
            help="concentration of von mises dist. used in rotate",
        )
        loader.add_argument(
            "--sc_stdv",
            default=0.12,
            type=float,
            help="std deviation of log-normal dist. used in scale",
        )
        loader.add_argument(
            "--sh_kappa",
            default=20,
            type=float,
            help="concentration of von mises dist. used in shear",
        )
        loader.add_argument(
            "--trans_prob",
            default=0.5,
            type=float,
            help="probabiliti to perform a transformation",
        )
        loader.add_argument(
            "--do_prior",
            default=False,
            type=bool,
            help="Compute prior distribution over classes",
        )
        # ----------------------------------------------------------------------
        # ----- Define NN parameters
        # ----------------------------------------------------------------------
        net = self.parser.add_argument_group("Neural Networks Parameters")
        net.add_argument(
            "--input_channels",
            default=3,
            type=int,
            help="Number of channels of input data",
        )
        # net.add_argument('--output_channels', default=2, type=int,
        #                 help="""Number of channels of labels.
        #                         If =1 then only lines will be extracted.""")
        net.add_argument(
            "--out_mode",
            default="LR",
            type=str,
            choices=["L", "R", "LR"],
            help="""Type of output:
                        L: Only Text Lines will be extracted
                        R: Only Regions will be extracted
                        LR: Lines and Regions will be extracted""",
        )
        net.add_argument(
            "--cnn_ngf", default=64, type=int, help="Number of filters of CNNs"
        )
        n_meg = net.add_mutually_exclusive_group(required=False)
        n_meg.add_argument(
            "--use_gan",
            dest="use_gan",
            action="store_true",
            help="USE GAN to compute G loss",
        )
        n_meg.add_argument(
            "--no-use_gan",
            dest="use_gan",
            action="store_false",
            help="do not use GAN to compute G loss",
        )
        n_meg.set_defaults(use_gan=True)
        net.add_argument(
            "--gan_layers", default=3, type=int, help="Number of layers of GAN NN"
        )
        net.add_argument(
            "--loss_lambda",
            default=100,
            type=float,
            help="Lambda weith to copensate GAN vs G loss",
        )
        net.add_argument(
            "--g_loss",
            default="L1",
            type=str,
            choices=["L1", "MSE", "smoothL1"],
            help="Loss function for G NN",
        )
        net.add_argument(
            "--net_out_type",
            default="C",
            type=str,
            choices=["C", "R"],
            help="""Compute the problem as a classification or Regresion:
                         C: Classification
                         R: Regresion""",
        )
        # ----------------------------------------------------------------------
        # ----- Define Optimizer parameters
        # ----------------------------------------------------------------------
        optim = self.parser.add_argument_group("Optimizer Parameters")
        optim.add_argument(
            "--adam_lr",
            default=0.001,
            type=float,
            help="Initial Lerning rate for ADAM opt",
        )
        optim.add_argument(
            "--adam_beta1",
            default=0.5,
            type=float,
            help="First ADAM exponential decay rate",
        )
        optim.add_argument(
            "--adam_beta2",
            default=0.999,
            type=float,
            help="Secod ADAM exponential decay rate",
        )
        # ----------------------------------------------------------------------
        # ----- Define Train parameters
        # ----------------------------------------------------------------------
        train = self.parser.add_argument_group("Training Parameters")
        tr_meg = train.add_mutually_exclusive_group(required=False)
        tr_meg.add_argument(
            "--do_train", dest="do_train", action="store_true", help="Run train stage"
        )
        tr_meg.add_argument(
            "--no-do_train",
            dest="do_train",
            action="store_false",
            help="Do not run train stage",
        )
        tr_meg.set_defaults(do_train=True)
        train.add_argument(
            "--cont_train",
            default=False,
            action="store_true",
            help="Continue training using this model",
        )
        train.add_argument(
            "--prev_model",
            default=None,
            type=str,
            help="Use this previously trainned model",
        )
        train.add_argument(
            "--save_rate",
            default=10,
            type=int,
            help="Save checkpoint each --save_rate epochs",
        )
        train.add_argument(
            "--tr_data",
            default="./data/train/",
            type=str,
            help="""Train data folder. Train images are
                                   expected there, also PAGE XML files are
                                   expected to be in --tr_data/page folder""",
        )
        train.add_argument(
            "--epochs", default=100, type=int, help="Number of training epochs"
        )
        train.add_argument(
            "--tr_img_list",
            default="",
            type=str,
            help="""List to all images ready to be used by NN
                                   train, if not provide it will be generated from
                                   original data.""",
        )
        train.add_argument(
            "--tr_label_list",
            default="",
            type=str,
            help="""List to all label ready to be used by NN
                                   train, if not provide it will be generated from
                                   original data.""",
        )
        train.add_argument(
            "--fix_class_imbalance",
            default=True,
            type=bool,
            help="use weights at loss function to handle class imbalance.",
        )
        train.add_argument(
            "--weight_const",
            default=1.02,
            type=float,
            help="weight constant to fix class imbalance",
        )
        # ----------------------------------------------------------------------
        # ----- Define Test parameters
        # ----------------------------------------------------------------------
        test = self.parser.add_argument_group("Test Parameters")
        te_meg = test.add_mutually_exclusive_group(required=False)
        te_meg.add_argument(
            "--do_test", dest="do_test", action="store_true", help="Run test stage"
        )
        te_meg.add_argument(
            "--no-do_test",
            dest="do_test",
            action="store_false",
            help="Do not run test stage",
        )
        te_meg.set_defaults(do_test=False)
        test.add_argument(
            "--te_data",
            default="./data/test/",
            type=str,
            help="""Test data folder. Test images are
                                 expected there, also PAGE XML files are
                                 expected to be in --te_data/page folder
                                 """,
        )
        test.add_argument(
            "--te_img_list",
            default="",
            type=str,
            help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """,
        )
        test.add_argument(
            "--te_label_list",
            default="",
            type=str,
            help="""List to all label ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """,
        )
        test.add_argument(
            "--do_off",
            default=True,
            type=bool,
            help="Turn DropOut Off during inference",
        )
        # ----------------------------------------------------------------------
        # ----- Define Validation parameters
        # ----------------------------------------------------------------------
        validation = self.parser.add_argument_group("Validation Parameters")
        v_meg = validation.add_mutually_exclusive_group(required=False)
        v_meg.add_argument(
            "--do_val", dest="do_val", action="store_true", help="Run Validation stage"
        )
        v_meg.add_argument(
            "--no-do_val",
            dest="do_val",
            action="store_false",
            help="do not run Validation stage",
        )
        v_meg.set_defaults(do_val=False)
        validation.add_argument(
            "--val_data",
            default="./data/val/",
            type=str,
            help="""Validation data folder. Validation images are
                                 expected there, also PAGE XML files are
                                 expected to be in --te_data/page folder
                                 """,
        )
        validation.add_argument(
            "--val_img_list",
            default="",
            type=str,
            help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """,
        )
        validation.add_argument(
            "--val_label_list",
            default="",
            type=str,
            help="""List to all label ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """,
        )
        # ----------------------------------------------------------------------
        # ----- Define Production parameters
        # ----------------------------------------------------------------------
        production = self.parser.add_argument_group("Production Parameters")
        p_meg = production.add_mutually_exclusive_group(required=False)
        p_meg.add_argument(
            "--do_prod",
            dest="do_prod",
            action="store_true",
            help="Run production stage",
        )
        p_meg.add_argument(
            "--no-do_prod",
            dest="do_prod",
            action="store_false",
            help="Do not run production stage",
        )
        p_meg.set_defaults(do_prod=False)
        production.add_argument(
            "--prod_data",
            default="./data/prod/",
            type=str,
            help="""Production data folder. Production images are
                                 expected there.
                                 """,
        )
        production.add_argument(
            "--prod_img_list",
            default="",
            type=str,
            help="""List to all images ready to be used by NN
                                 train, if not provide it will be generated from
                                 original data.
                                 """,
        )
        # ----------------------------------------------------------------------
        # ----- Define Evaluation parameters
        # ----------------------------------------------------------------------
        evaluation = self.parser.add_argument_group("Evaluation Parameters")
        evaluation.add_argument(
            "--target_list",
            default="",
            type=str,
            help="List of ground-truth PAGE-XML files",
        )
        evaluation.add_argument(
            "--hyp_list", default="", type=str, help="List of hypotesis PAGE-XMLfiles"
        )

    def _convert_file_to_args(self, arg_line):
        return arg_line.split(" ")

    def _str_to_bool(self, data):
        """
        Nice way to handle bool flags:
        from: https://stackoverflow.com/a/43357954
        """
        if data.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif data.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def _check_out_dir(self, pointer):
        """ Checks if the dir is wirtable"""
        if os.path.isdir(pointer):
            # --- check if is writeable
            if os.access(pointer, os.W_OK):
                if not (os.path.isdir(pointer + "/checkpoints")):
                    os.makedirs(pointer + "/checkpoints")
                    self.logger.debug(
                        "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                    )
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not writeable.".format(pointer)
                )
        else:
            try:
                os.makedirs(pointer)
                self.logger.debug("Creating output dir: {}".format(pointer))
                os.makedirs(pointer + "/checkpoints")
                self.logger.debug(
                    "Creating checkpoints dir: {}".format(pointer + "/checkpoints")
                )
                return pointer
            except OSError as e:
                raise argparse.ArgumentTypeError(
                    "{} folder does not exist and cannot be created\n{}".format(e)
                )

    def _check_in_dir(self, pointer):
        """check if path exists and is readable"""
        if os.path.isdir(pointer):
            if os.access(pointer, os.R_OK):
                return pointer
            else:
                raise argparse.ArgumentTypeError(
                    "{} folder is not readable.".format(pointer)
                )
        else:
            raise argparse.ArgumentTypeError(
                "{} folder does not exists".format(pointer)
            )

    def _check_to_int_array(self, data):
        """check is size is 256 multiple"""
        data = int(data)
        if data > 0 and data % 256 == 0:
            return data
        else:
            raise argparse.ArgumentTypeError(
                "Image size must be multiple of 256: {} is not".format(data)
            )

    def _build_class_regions(self):
        """given a list of regions assign a equaly separated class to each one"""
        class_dic = OrderedDict()
        # --- for classification keep regions as a seq of intigers
        if self.opts.do_class:
            for c, r in enumerate(self.opts.regions):
                class_dic[r] = c + 1
            return class_dic
        # --- for regresion put all values equally separated in the cont
        # --- line from 0 to 255
        n_class = len(self.opts.regions)
        # --- div by n_claass + 1 because background is another "class"
        class_gap = int(256 / n_class + 1)
        class_id = int(class_gap / 2) + class_gap
        for c in self.opts.regions:
            class_dic[c] = class_id
            class_id = class_id + class_gap

        return class_dic

    def _build_merged_regions(self):
        """build dic of regions to be merged into a single class"""
        if self.opts.merge_regions == None:
            return None
        to_merge = {}
        msg = ""
        for c in self.opts.merge_regions:
            try:
                parent, childs = c.split(":")
                if parent in self.opts.regions:
                    to_merge[parent] = childs.split(",")
                else:
                    msg = '\nRegion "{}" to merge is not defined as region'.format(
                        parent
                    )
                    raise
            except:
                raise argparse.ArgumentTypeError(
                    "Malformed argument {}".format(c) + msg
                )

        return to_merge

    def _build_region_types(self):
        """ build a dic of regions and their respective type"""
        reg_type = {"full_page": "TextRegion"}
        if self.opts.region_type == None:
            for reg in self.opts.regions:
                reg_type[reg] = "TextRegion"
            return reg_type
        msg = ""
        for c in self.opts.region_type:
            try:
                parent, childs = c.split(":")
                regs = childs.split(",")
                for reg in regs:
                    if reg in self.opts.regions:
                        reg_type[reg] = parent
                    else:
                        msg = '\nCannot assign region "{0}" to any type. {0} not defined as region'.format(
                            reg
                        )
            except:
                raise argparse.ArgumentTypeError(
                    "Malformed argument {}".formatt(c) + msg
                )
        return reg_type

    def _define_output_channels(self):
        if self.opts.net_out_type == "C":
            if self.opts.out_mode == "L":
                n_ch = 2
            elif self.opts.out_mode == "R":
                n_ch = 1 + len(self.opts.regions)
            elif self.opts.out_mode == "LR":
                n_ch = 3 + len(self.opts.regions)
            else:
                raise argparse.ArgumentTypeError("Malformed argument --out_mode")
            return n_ch
        if self.opts.net_out_type == "R":
            if self.opts.out_mode == "L" or self.opts.out_mode == "R":
                n_ch = 1
            elif self.opts.out_mode == "LR":
                n_ch = 2
            else:
                raise argparse.ArgumentTypeError("Malformed argument --out_mode")
            return n_ch

    def shortest_arg(self, arg):
        """
        search for the shortest valid argument using levenshtein edit distance
        """
        d = {key: (1000, key) for key in arg}
        for k in vars(self.opts):
            for t in arg:
                l = levenshtein(k, t)
                if l < d[t][0]:
                    d[t] = (l, k)
        return ["--" + d[k][1] for k in arg]

    def parse(self):
        """Perform arguments parsing"""
        # --- Parse initialization + command line arguments
        # --- Arguments priority stack:
        # ---    1) command line arguments
        # ---    2) config file arguments
        # ---    3) default arguments
        self.opts, unkwn = self.parser.parse_known_args()
        if unkwn:
            msg = "unrecognized command line arguments: {}\n".format(unkwn)
            msg += "do you mean: {}\n".format(self.shortest_arg(unkwn))
            msg += "In the meanwile, solve this maze:\n"
            msg += art.make_maze()
            self.parser.error(msg)

        # --- Parse config file if defined
        if self.opts.config != None:
            self.logger.info("Reading configuration from {}".format(self.opts.config))
            self.opts, unkwn_conf = self.parser.parse_known_args(
                ["@" + self.opts.config], namespace=self.opts
            )
            if unkwn_conf:
                msg = "unrecognized  arguments in config file: {}\n".format(unkwn_conf)
                msg += "do you mean: {}\n".format(self.shortest_arg(unkwn_conf))
                msg += "In the meanwile, solve this maze:\n"
                msg += art.make_maze()
                self.parser.error(msg)
            self.opts = self.parser.parse_args(namespace=self.opts)
        # --- Preprocess some input variables
        # --- enable/disable
        self.opts.use_gpu = self.opts.gpu != -1
        # --- make sure to don't use pinned memory when CPU only, DataLoader class
        # --- will copy tensors into GPU by default if pinned memory is True.
        if not self.opts.use_gpu:
            self.opts.pin_memory = False
        # --- set logging data
        self.opts.log_level_id = getattr(logging, self.opts.log_level.upper())
        self.opts.log_file = self.opts.work_dir + "/" + self.opts.exp_name + ".log"
        # --- build classes
        self.opts.do_class = self.opts.net_out_type == "C"
        self.opts.regions_colors = self._build_class_regions()
        self.opts.merged_regions = self._build_merged_regions()
        self.opts.region_types = self._build_region_types()
        # --- add merde regions to color dic, so parent and merged will share the same color
        for parent, childs in self.opts.merged_regions.items():
            for child in childs:
                self.opts.regions_colors[child] = self.opts.regions_colors[parent]

        # --- TODO: Move this create dir to check inputs function
        self._check_out_dir(self.opts.work_dir)
        self.opts.checkpoints = os.path.join(self.opts.work_dir, "checkpoints/")
        # if self.opts.do_class:
        #    self.opts.line_color = 1
        # --- define network output channels based on inputs
        self.opts.output_channels = self._define_output_channels()

        return self.opts

    def __str__(self):
        """pretty print handle"""
        data = "------------ Options -------------"
        try:
            for k, v in sorted(vars(self.opts).items()):
                data = data + "\n" + "{0:15}\t{1}".format(k, v)
        except:
            data = data + "\nNo arguments parsed yet..."

        data = data + "\n---------- End  Options ----------\n"
        return data

    def __repr__(self):
        return self.__str__()
