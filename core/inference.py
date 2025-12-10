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
from mmv_im2im.configs.config_base import ProgramConfig, configuration_validation, parse_adaptor
from mmv_im2im.map_extractor import MapExtractor
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from core.utils import topology_preserving_thinning
from dataclasses import dataclass
from pyrallis import field
import argparse
import dataclasses
import sys
from argparse import HelpFormatter, Namespace
from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Sequence, Text, Type, Union, TypeVar, Generic, Optional
from pyrallis import utils, cfgparsing
from pyrallis.help_formatter import SimpleHelpFormatter
from pyrallis.parsers import decoding
from pyrallis.utils import Dataclass, PyrallisException
from pyrallis.wrappers import DataclassWrapper
from tqdm.notebook import tqdm
import gc
import torch
import bioio_tifffile
from ipyfilechooser import FileChooser

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
    
    #  YAML config file path
    yaml_path_widget = FileChooser(
        
        Path.cwd().as_posix(),
        title='Select the YAML configuration file',
        select_default=False 
    )
    yaml_path_widget.filter_pattern = ['*.yaml'] 
    yaml_path_widget.layout = widgets.Layout(width='80%')

    ckpt_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Selcet the CKPT model weights file ',
        select_default=False 
    )
    ckpt_path_widget.filter_pattern = ['*.ckpt']
    ckpt_path_widget.layout = widgets.Layout(width='80%')

   
    path_base_widget = FileChooser(
        Path.cwd().as_posix(), 
        title='Select the root of the split_3d images folder',
        select_default=False 
    )
    path_base_widget.show_only_dirs = True 
    path_base_widget.layout = widgets.Layout(width='80%')

    # --- Inference Options --- section
    inference_header = widgets.Label(
        value="--- Inference Options ---",
        style={'font_weight': 'bold'}
    )

    # Checkbox for Use Max projection
    multi_pred_mode_dropdown = widgets.Dropdown(
        options={
            'single': 'single', 
            'maximum': 'max',
            'mean': 'mean',
            'variance': 'var'
        },
        value='single', 
        description='Prediction mode:',
        disabled=False,
        style={'description_width':'50%'} 
    )
    
    n_samples_text = widgets.Text(
        value='1', # Valor inicial por defecto (para 'single')
        description='N Samples:',
        disabled=True, # Inicialmente deshabilitado porque 'single' debe ser 1
        layout=widgets.Layout(width='50%')
    )

    use_max_proj_checkbox = widgets.Checkbox(
        value=False,
        description="Use Max projection",
        disabled=False
    )

    # Postprocessing section header
    postprocessing_header = widgets.Label(
        value="--- Postprocessing Options ---",
        style={'font_weight': 'bold'}
    )
    # Default parameters
    default_params = {
        "pixelDimensions": "1.0, 1.0, 1.0"
    }

    meta_pixel_dim_widget = widgets.Checkbox(
        value=True,
        description='Use pixel dimensions from image metadata',
        indent=True,
         style={'description_width': '0%'}
    )
    
    # Create widgets for each parameter
    pixel_dim_widget = widgets.Text(
        value=default_params["pixelDimensions"],
        description='Pixel Dimensions(Z,Y,X):',
        layout=widgets.Layout(description_width='initial'),
        style={'description_width': '60%'}
    )
    # Checkboxes for post-processing

    apply_thickness_checkbox = widgets.Checkbox(
        value=True,
        description="Apply Thickness Adjustment",
        disabled=False
    )
    min_thickness_list_text = widgets.Text(
        value="1000,2", 
        placeholder="e.g., 1000, 1",
        description="Min Thicknesses :",
        disabled=False,
        layout=widgets.Layout(width='80%')
    )

    # Checkbox and text input for Remove Small Objects
    use_remove_objects_checkbox = widgets.Checkbox(
        value=True,
        description="Apply Remove Small Objects",
        disabled=False
    )
    remove_objects_text = widgets.Text(
        value="50,30", 
        placeholder="e.g., 50, 100",
        description="Sizes :",
        disabled=False,
        layout=widgets.Layout(width='80%')
    )
    apply_holes_checkbox = widgets.Checkbox(
        value=True,
        description="Apply Small Holes Correction",
        disabled=False
    )

    hole_size_threshold_text = widgets.Text(
        value="90,90", # Default value for the original implementation
        placeholder="e.g., 15, 30",
        description="Hole Thresholds :",
        disabled=False,
        layout=widgets.Layout(width='80%')
    )

    apply_pericytes_checkbox = widgets.Checkbox(
        value=True,
        description="Apply Pericytes Correction",
        disabled=False,
        indent=False
    )

    def toggle_thickness_text(change):
        min_thickness_list_text.layout.display = 'block' if change['new'] else 'none'

    def toggle_remove_objects_text(change):
        remove_objects_text.layout.display = 'block' if change['new'] else 'none'

    def toggle_hole_threshold_text(change):
        hole_size_threshold_text.layout.display = 'block' if change['new'] else 'none'
    
    def update_n_samples_on_mode_change(change):
        if change['new'] == 'single':
            n_samples_text.value = '1'
            n_samples_text.disabled = True
        else:
            if n_samples_text.value == '1':
                 n_samples_text.value = '10' 
            n_samples_text.disabled = False
    
    multi_pred_mode_dropdown.observe(update_n_samples_on_mode_change, names='value')

    min_thickness_list_text.layout.display = 'block' if apply_thickness_checkbox.value else 'none'
    remove_objects_text.layout.display = 'block' if use_remove_objects_checkbox.value else 'none'
    hole_size_threshold_text.layout.display = 'block' if apply_holes_checkbox.value else 'none' 

    apply_thickness_checkbox.observe(toggle_thickness_text, names='value')
    use_remove_objects_checkbox.observe(toggle_remove_objects_text, names='value')
    apply_holes_checkbox.observe(toggle_hole_threshold_text, names='value')

    run_button = widgets.Button(description="Run Inference" , button_style='success', )
    output = widgets.Output()

    display(yaml_path_widget, ckpt_path_widget, path_base_widget, 
                inference_header, multi_pred_mode_dropdown, n_samples_text, use_max_proj_checkbox,
                postprocessing_header,
                pixel_dim_widget,meta_pixel_dim_widget,  
                use_remove_objects_checkbox, remove_objects_text,
                apply_holes_checkbox, hole_size_threshold_text,
                apply_thickness_checkbox, min_thickness_list_text,apply_pericytes_checkbox, 
                run_button, output)
    
    def parse_int_list(text_value):
        """Converts a comma-separated string of numbers to a list of integers."""
        if not text_value.strip():
            return None
        try:
            return [int(x.strip()) for x in text_value.split(',') if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid list of numbers: '{text_value}'. Ensure all entries are integers separated by commas.")

    def on_button_clicked(b):
        with output:
            output.clear_output()
            selected_yaml = yaml_path_widget.selected
            selected_ckpt = ckpt_path_widget.selected
            selected_path_base = path_base_widget.selected
            
            apply_max_proj = use_max_proj_checkbox.value
            apply_thickness = apply_thickness_checkbox.value
            apply_remove_objects = use_remove_objects_checkbox.value
            apply_holes = apply_holes_checkbox.value
            selected_multi_pred_mode = multi_pred_mode_dropdown.value
            selected_n_samples = n_samples_text.value

        
            pixel_dim = (1,1,1)
            if meta_pixel_dim_widget.value:
                pixel_dim = 'auto'
            else:
                pixel_dim = tuple(float(d.strip()) for d in pixel_dim_widget.value.split(','))

            if not selected_yaml or not selected_ckpt or not selected_path_base:
                raise ValueError("Please select correct input files (ckpt/yaml/3Dimages)")
                return

            try:
                n_samples_int = int(selected_n_samples)
                if n_samples_int <= 0:
                    raise ValueError(f"Unexpected Value {n_samples_int}. Provide a positive integer.")
            except ValueError as e:
                print(f"ERROR: {e}.")
                return

            min_thickness_list_values = None
            if apply_thickness:
                try:
                    min_thickness_list_values = parse_int_list(min_thickness_list_text.value)
                    if min_thickness_list_values is None:
                        raise ValueError("Thickness Adjustment is enabled but no minimum thicknesses were provided.")
                except ValueError as e:
                    print(f"Error in Min Thicknesses list: {e}")
                    return
            
            remove_objects_sizes = None
            if apply_remove_objects:
                try:
                    remove_objects_sizes = parse_int_list(remove_objects_text.value)
                    if remove_objects_sizes is None:
                        raise ValueError("Remove Small Objects is enabled but no sizes were provided.")
                except ValueError as e:
                    print(f"Error in Remove Small Objects sizes: {e}")
                    return

            hole_size_thresholds = None
            if apply_holes:
                try:
                    hole_size_thresholds = parse_int_list(hole_size_threshold_text.value)
                    if hole_size_thresholds is None:
                        raise ValueError("Hole Correction is enabled but no sizes were provided.")
                except ValueError as e:
                    print(f"Error in Hole Correction Correction thresholds: {e}")
                    return

            try:
                #read config
                cfg = parse_adaptor_jpnb(config_class=ProgramConfig, config=selected_yaml)
        
                cfg.mode = 'uncertainty_map'
                cfg.model.net['pred_slice2vol']['jupyter'] = True
                cfg.model.net['pred_slice2vol']['pixel_dim'] = pixel_dim
                cfg.model.net['pred_slice2vol']['multi_pred_mode'] = selected_multi_pred_mode
                cfg.model.net['pred_slice2vol']['n_samples'] = n_samples_int
                if cfg.model.net['func_name'] == 'ProbabilisticUNet':
                    cfg.model.net['pred_slice2vol']['n_class_correction'] = cfg.model.net['params']['n_classes']
                else:
                    cfg.model.net['pred_slice2vol']['n_class_correction'] = cfg.model.net['params']['out_channels']
                cfg.model.net['pred_slice2vol']['max_proj'] = apply_max_proj
                cfg.model.net['pred_slice2vol']['remove_object_size'] = remove_objects_sizes
                cfg.model.net['pred_slice2vol']['hole_size_threshold'] = hole_size_thresholds
                if hole_size_thresholds:
                    cfg.model.net['pred_slice2vol']['sv'] = True    
                cfg.model.net['pred_slice2vol']['min_thickness_list'] = min_thickness_list_values
                cfg.model.net['pred_slice2vol']['perycites_correction'] = apply_pericytes_checkbox.value    
                cfg.model.checkpoint = Path(selected_ckpt)
                cfg.data.inference_output.path = Path(selected_path_base) / Path("model_predictions")
                cfg.data.inference_input.dir = Path(selected_path_base) / Path("split_3d")
                cfg.data.inference_input.data_type = ".tiff,.tif"
                cfg = configuration_validation(cfg)
                cfg.data.inference_output.path.mkdir(parents=True, exist_ok=True)
                
                #generate inference class
                exe = MapExtractor(cfg)
                exe.run_inference()
                print("########################Predictions Ready#############################")
            except Exception as e:
                print(f"An error occurred: {e}")
    run_button.on_click(on_button_clicked)
