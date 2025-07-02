import os
# Suppress specific numpy DeprecationWarning
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:numpy"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='monai')
warnings.filterwarnings("ignore", category=UserWarning, message='^In the future `np.bool` will be defined as the corresponding NumPy scalar')
warnings.filterwarnings("ignore", category=DeprecationWarning, message='^In the future `np.bool` will be defined as the corresponding NumPy scalar')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')

from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from mmv_im2im.configs.config_base import ProgramConfig, configuration_validation
from mmv_im2im import ProjectTester
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from core.utils import topology_preserving_thinning
from dataclasses import dataclass
from pathlib import Path
from pyrallis import field
import argparse
import dataclasses
import sys
import warnings
from argparse import HelpFormatter, Namespace
from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Sequence, Text, Type, Union, TypeVar, Generic, Optional
from pyrallis import utils, cfgparsing
from pyrallis.help_formatter import SimpleHelpFormatter
from pyrallis.parsers import decoding
from pyrallis.utils import Dataclass, PyrallisException
from pyrallis.wrappers import DataclassWrapper


logger = getLogger(__name__)

T = TypeVar("T")


class ArgumentParser(Generic[T], argparse.ArgumentParser):
    def __init__(
        self,
        config_class: Type[T],
        config: Optional[str] = None,
        formatter_class: Type[HelpFormatter] = SimpleHelpFormatter,
        *args,
        **kwargs,
    ):
        kwargs["formatter_class"] = formatter_class
        super().__init__(*args, **kwargs)

        self.constructor_arguments: Dict[str, Dict] = defaultdict(dict)

        self._wrappers: List[DataclassWrapper] = []

        self.config = config
        self.config_class = config_class

        self._assert_no_conflicts()
        self.add_argument(
            f"--{utils.CONFIG_ARG}",
            type=str,
            help="Path for a config file to parse with pyrallis",
        )
        self.set_dataclass(config_class)

    def set_dataclass(
        self,
        dataclass: Union[Type[Dataclass], Dataclass],
        prefix: str = "",
        default: Union[Dataclass, Dict] = None,
        dataclass_wrapper_class: Type[DataclassWrapper] = DataclassWrapper,
    ):
        if not isinstance(dataclass, type):
            default = dataclass if default is None else default
            dataclass = type(dataclass)

        new_wrapper = dataclass_wrapper_class(dataclass, prefix=prefix, default=default)
        self._wrappers.append(new_wrapper)
        self._wrappers += new_wrapper.descendants

        for wrapper in self._wrappers:
            logger.debug(
                f"Adding arguments for dataclass: {wrapper.dataclass} "
                f"at destination {wrapper.dest}"
            )
            wrapper.add_arguments(parser=self)

    def _assert_no_conflicts(self):
        if utils.CONFIG_ARG in [
            field.name for field in dataclasses.fields(self.config_class)
        ]:
            raise PyrallisException(
                f"{utils.CONFIG_ARG} is a reserved word for pyrallis"
            )

    def parse_args(self, args=None, namespace=None) -> T:
        return super().parse_args(args, namespace)

    def parse_known_args(
        self,
        args: Sequence[Text] = None,
        namespace: Namespace = None,
        attempt_to_reorder: bool = False,
    ):
        if args is None:
            args = sys.argv[1:]
        else:
            args = list(args)

        if "--help" not in args:
            for action in self._actions:
                action.default = argparse.SUPPRESS
                action.type = str
        parsed_args, unparsed_args = super().parse_known_args(args, namespace)

        parsed_args = self._postprocessing(parsed_args)
        return parsed_args, unparsed_args

    def print_help(self, file=None):
        return super().print_help(file)

    def _postprocessing(self, parsed_args: Namespace) -> T:
        logger.debug("\nPOST PROCESSING\n")
        logger.debug(f"(raw) parsed args: {parsed_args}")

        parsed_arg_values = vars(parsed_args)

        for key in parsed_arg_values:
            parsed_arg_values[key] = cfgparsing.parse_string(parsed_arg_values[key])

        config = self.config

        if utils.CONFIG_ARG in parsed_arg_values:
            new_config = parsed_arg_values[utils.CONFIG_ARG]
            if config is not None:
                warnings.warn(
                    UserWarning(f"Overriding default {config} with {new_config}")
                )
            if Path(new_config).is_file():
                config = new_config
            else:
                new_config = str(new_config)
                print(f"trying to locate preset config for {new_config} ...")

                config = Path(__file__).parent / f"preset_{new_config}.yaml"
            del parsed_arg_values[utils.CONFIG_ARG]

        if config is not None:
            print(f"loading configuration from {config} ...")
            file_args = cfgparsing.load_config(open(config, "r"))
            file_args = utils.flatten(file_args, sep=".")
            file_args.update(parsed_arg_values)
            parsed_arg_values = file_args
            print("configuration loading is completed")

        deflat_d = utils.deflatten(parsed_arg_values, sep=".")
        cfg = decoding.decode(self.config_class, deflat_d)

        return cfg

def parse_adaptor_jpnb(
    config_class: Type[T],
    config: Optional[Union[Path, str]] = None,
    args: Optional[Sequence[str]] = None,
) -> T:
    parser = ArgumentParser(config_class=config_class, config=config)
    return parser.parse_args(args=[])

def create_inference_menu():
    # Text input for YAML config file path
    yaml_path_text = widgets.Text(
        value="./semantic_seg_2d_inference_2class.yaml", # Default value
        placeholder="Path to YAML file ",
        description="YAML Path:",
        disabled=False,
        layout=widgets.Layout(width='80%') # Make it wider for paths
    )

    # Text input for Checkpoint file path
    ckpt_path_text = widgets.Text(
        value="./version_2023_06.ckpt", # Default value
        placeholder="Weights Path",
        description="CKPT Path:",
        disabled=False,
        layout=widgets.Layout(width='80%') # Make it wider for paths
    )

    # Text input for the base path
    path_base_text = widgets.Text(
        value="/path/to/splitted/3d/files",
        placeholder="Input path",
        description="Input base path:",
        disabled=False,
        layout=widgets.Layout(width='80%'),
        style={'description_width': '12%'} 
    )

    run_button = widgets.Button(description="Run Inference" , button_style='success', )
    output = widgets.Output()

    display(yaml_path_text, ckpt_path_text, path_base_text, run_button, output)

    def on_button_clicked(b):
        with output:
            output.clear_output()
            selected_yaml = yaml_path_text.value
            selected_ckpt = ckpt_path_text.value
            selected_path_base = path_base_text.value

            try:
                # Configuration for the model
                cfg = parse_adaptor_jpnb(config_class=ProgramConfig, config=selected_yaml)
                cfg = configuration_validation(cfg)
                cfg.model.checkpoint = Path(selected_ckpt)

                # Define the executor for inference
                executor = ProjectTester(cfg)
                executor.setup_model()
                executor.setup_data_processing()

                # Get the data, run inference, and save the result
                path_base = Path(selected_path_base)
                input_path = path_base / Path("split_3d")
                out_p = path_base / Path("pred_2class")
                out_p.mkdir(parents=True, exist_ok=True)

                filenames = sorted(input_path.glob("*.tiff"))

                num = 0
                for fn in filenames:
                    num = num + 1
                    print(f"--Predicting: {num}/{len(filenames)} ...")
                    img = BioImage(fn).get_image_data("CZYX", T=0)

                    out_list = []
                    for zz in range(img.shape[1]):
                        im_input = img[:, zz, :, :]
                        seg = executor.process_one_image(im_input)
                        out_list.append(np.squeeze(seg))
                    seg_full = np.stack(out_list, axis=0)

                    # Attempt to remove pericytes by post-processing
                    seg_2 = remove_small_objects(seg_full == 2, min_size=64)
                    seg_2_mid = np.logical_xor(seg_2, remove_small_objects(seg_2, min_size=300))
                    for zz in range(seg_2_mid.shape[0]):
                        seg_label, num_obj = label(seg_2_mid[zz, :, :], return_num=True)
                        if num_obj > 0:
                            stats = regionprops(seg_label)
                            for ii in range(num_obj):
                                if stats[ii].eccentricity < 0.88 and stats[ii].solidity > 0.85 and stats[ii].area < 150:
                                    seg_z = seg_2[zz, :, :]
                                    seg_z[seg_label == (ii+1)] = 0
                                    seg_2[zz, :, :] = seg_z

                    seg_full[seg_full == 2] = 1
                    seg_full[seg_2 > 0] = 2

                    # Another minor fix: remove small holes due to segmentation errors
                    hole_size_threshold = 15
                    seg_1 = remove_small_objects(seg_full==1, min_size=50)
                    seg_2 = seg_full == 2
                    for zz in range(seg_full.shape[0]):
                        s_v = remove_small_holes(seg_1[zz, :, :], area_threshold=hole_size_threshold)
                        seg_1[zz, :, :] = s_v[:, :]

                        a_v = remove_small_holes(seg_2[zz, :, :], area_threshold=hole_size_threshold)
                        seg_2[zz, :, :] = a_v[:, :]

                    # Thickness adjustment
                    seg_string = topology_preserving_thinning(seg_full == 2, min_thickness=1, thin=1)
                    seg_thin = np.zeros_like(seg_full)
                    seg_thin[seg_string > 0] = 2
                    seg_thin[seg_full == 1] = 1

                    out_fn = out_p / fn.name
                    OmeTiffWriter.save(seg_full, out_fn, dim_order="ZYX")
                print("Inference completed.")

            except Exception as e:
                print(f"An error occurred: {e}")

    run_button.on_click(on_button_clicked)
    