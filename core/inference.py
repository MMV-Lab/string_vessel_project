import os
# Suppress specific numpy DeprecationWarning
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:numpy"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='monai')
warnings.filterwarnings("ignore", category=UserWarning, message='^In the future `np.bool` will be defined as the corresponding NumPy scalar')
warnings.filterwarnings("ignore", category=DeprecationWarning, message='^In the future `np.bool` will be defined as the corresponding NumPy scalar')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')
warnings.filterwarnings("ignore",category=FutureWarning)

import copy
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
from mmv_im2im.utils.utils import topology_preserving_thinning
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
    
    # 1. Pipeline Mode Selection
    pipeline_mode_header = widgets.Label(
        value="--- Pipeline mode ---",
        style={'font_weight': 'bold'}
    )
    
    pipeline_mode_dropdown = widgets.Dropdown(
        options=['single model', 'multi model'],
        value='single model',
        description='Mode:',
        disabled=False,
        style={'description_width': 'initial'}
    )

    # 2. Config Options Header
    config_header = widgets.Label(
        value="--- Config Options ---",
        style={'font_weight': 'bold'}
    )

    # YAML config file path
    yaml_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select the YAML configuration file',
        select_default=False 
    )
    yaml_path_widget.filter_pattern = ['*.yaml'] 
    yaml_path_widget.layout = widgets.Layout(width='80%')

    # --- SINGLE MODEL WIDGETS ---
    ckpt_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select the CKPT model weights file',
        select_default=False 
    )
    ckpt_path_widget.filter_pattern = ['*.ckpt']
    ckpt_path_widget.layout = widgets.Layout(width='80%')

    # --- MULTI MODEL WIDGETS ---
    models_folder_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select the training models folder',
        select_default=False
    )
    models_folder_widget.show_only_dirs = True
    models_folder_widget.layout = widgets.Layout(width='80%', display='none')

    output_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select predictions output',
        select_default=False
    )
    output_path_widget.show_only_dirs = True
    output_path_widget.layout = widgets.Layout(width='80%', display='none')

    # Sub-options for Multi-Model
    weight_options_dropdown = widgets.Dropdown(
        options=['last', 'min', 'custom'], 
        value='last',
        description='Weight Options:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(display='none', width='50%')
    )

    # Container for dynamic weights (One selector per subfolder)
    custom_weights_container = widgets.VBox(
        layout=widgets.Layout(display='none', margin='10px 0px 10px 20px')
    )
    
    # Input Images
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
        value='1',
        description='N Samples:',
        disabled=True,
        layout=widgets.Layout(width='50%')
    )

    use_max_proj_checkbox = widgets.Checkbox(
        value=False,
        description="Use Max projection",
        disabled=False
    )

    # Postprocessing section
    postprocessing_header = widgets.Label(
        value="--- Postprocessing Options ---",
        style={'font_weight': 'bold'}
    )
    default_params = {"pixelDimensions": "1.0, 1.0, 1.0"}

    meta_pixel_dim_widget = widgets.Checkbox(
        value=True,
        description='Use pixel dimensions from image metadata',
        indent=True,
         style={'description_width': '0%'}
    )
    
    pixel_dim_widget = widgets.Text(
        value=default_params["pixelDimensions"],
        description='Pixel Dimensions(Z,Y,X):',
        layout=widgets.Layout(description_width='initial'),
        style={'description_width': '60%'}
    )

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
        value="90,90",
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

    # --- LOGIC FOR UI INTERACTIVITY ---

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
            
    def check_folder_structure():
        if not models_folder_widget.selected:
            return None
        path = Path(models_folder_widget.selected)
        if not path.exists():
            return None
        if list(path.glob('*.ckpt')):
            return "flat"
        subdirs = [x for x in path.iterdir() if x.is_dir()]
        for subdir in subdirs:
            if (subdir / "checkpoints").exists() and (subdir / "checkpoints").is_dir():
                return "nested"
        return "unknown"

    def populate_custom_weight_selectors():
        """New function to create specific dropdowns for each model subfolder"""
        if not models_folder_widget.selected:
            return
        
        models_path = Path(models_folder_widget.selected)
        subdirs = [x for x in models_path.iterdir() if x.is_dir() and (x / "checkpoints").exists()]
        
        # UI headers for the dynamic section
        new_children = [widgets.HTML(value="<b>Select .ckpt for each model:</b>")]
        
        for subdir in subdirs:
            ckpt_dir = subdir / "checkpoints"
            available_ckpts = sorted([f.name for f in ckpt_dir.glob("*.ckpt")])
            
            if not available_ckpts:
                new_children.append(widgets.Label(value=f"⚠️ No weights found in {subdir.name}/checkpoints"))
                continue
                
            # Create a dropdown for this specific folder
            dropdown = widgets.Dropdown(
                options=available_ckpts,
                value='last.ckpt' if 'last.ckpt' in available_ckpts else available_ckpts[0],
                description=f"{subdir.name}:",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='90%')
            )
            # Store the folder name directly in the widget for easy retrieval later
            dropdown.target_folder = subdir.name
            new_children.append(dropdown)
            
        custom_weights_container.children = tuple(new_children)

    def update_multimodel_ui(change=None):
        structure = check_folder_structure()
        if structure == "nested":
            weight_options_dropdown.layout.display = 'block'
            if weight_options_dropdown.value == 'custom':
                 custom_weights_container.layout.display = 'block'
                 populate_custom_weight_selectors()
            else:
                 custom_weights_container.layout.display = 'none'
        else:
            weight_options_dropdown.layout.display = 'none'
            custom_weights_container.layout.display = 'none'

    def toggle_pipeline_mode(change):
        mode = change['new']
        if mode == 'single model':
            ckpt_path_widget.layout.display = 'block'
            models_folder_widget.layout.display = 'none'
            output_path_widget.layout.display = 'none'
            weight_options_dropdown.layout.display = 'none'
            custom_weights_container.layout.display = 'none'
        else:
            ckpt_path_widget.layout.display = 'none'
            models_folder_widget.layout.display = 'block'
            output_path_widget.layout.display = 'block'
            update_multimodel_ui()

    # Observers
    multi_pred_mode_dropdown.observe(update_n_samples_on_mode_change, names='value')
    min_thickness_list_text.layout.display = 'block' if apply_thickness_checkbox.value else 'none'
    remove_objects_text.layout.display = 'block' if use_remove_objects_checkbox.value else 'none'
    hole_size_threshold_text.layout.display = 'block' if apply_holes_checkbox.value else 'none' 

    apply_thickness_checkbox.observe(toggle_thickness_text, names='value')
    use_remove_objects_checkbox.observe(toggle_remove_objects_text, names='value')
    apply_holes_checkbox.observe(toggle_hole_threshold_text, names='value')
    
    pipeline_mode_dropdown.observe(toggle_pipeline_mode, names='value')
    models_folder_widget.register_callback(update_multimodel_ui)
    weight_options_dropdown.observe(update_multimodel_ui, names='value')

    run_button = widgets.Button(description="Run Inference" , button_style='success', )
    output = widgets.Output()

    # UI Display
    display(
        pipeline_mode_header, pipeline_mode_dropdown,
        config_header,
        yaml_path_widget, 
        ckpt_path_widget,
        models_folder_widget,
        output_path_widget,
        weight_options_dropdown,
        custom_weights_container,
        path_base_widget, 
        inference_header, multi_pred_mode_dropdown, n_samples_text, use_max_proj_checkbox,
        postprocessing_header,
        pixel_dim_widget,meta_pixel_dim_widget,  
        use_remove_objects_checkbox, remove_objects_text,
        apply_holes_checkbox, hole_size_threshold_text,
        apply_thickness_checkbox, min_thickness_list_text,apply_pericytes_checkbox, 
        run_button, output
    )

    def parse_int_list(text_value):
        if not text_value.strip():
            return None
        try:
            return [int(x.strip()) for x in text_value.split(',') if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid list: '{text_value}'. Use integers separated by commas.")

    def run_single_inference(cfg_obj, ckpt_path, output_dir_path):
        try:
            print(f"--> Loading Weights: {Path(ckpt_path).name}")
            cfg_obj.model.checkpoint = Path(ckpt_path)
            cfg_obj.data.inference_output.path = Path(output_dir_path)
            cfg_obj.data.inference_output.path.mkdir(parents=True, exist_ok=True)
            
            cfg_obj = configuration_validation(cfg_obj)
            exe = MapExtractor(cfg_obj)
            exe.run_inference()
            
            del exe
            gc.collect()
            torch.cuda.empty_cache()
            print(f"--> Done: {output_dir_path}")
        except Exception as e:
            print(f"Error processing {ckpt_path}: {e}")

    def on_button_clicked(b):
        with output:
            output.clear_output()
            
            mode = pipeline_mode_dropdown.value
            selected_yaml = yaml_path_widget.selected
            selected_path_base = path_base_widget.selected
            
            if not selected_yaml or not selected_path_base:
                print("Error: Required paths missing.")
                return

            # Pixel Dim logic
            if meta_pixel_dim_widget.value:
                pixel_dim = 'auto'
            else:
                try:
                    pixel_dim = tuple(float(d.strip()) for d in pixel_dim_widget.value.split(','))
                except:
                    print("Error: Invalid Pixel Dimensions.")
                    return

            try:
                base_cfg = parse_adaptor_jpnb(config_class=ProgramConfig, config=selected_yaml)
                
                if 'spatial_dims' in  base_cfg.model.net['params'] : 
                    spatial_dims = base_cfg.model.net.get('params', {}).get('spatial_dims', 2)
                else:
                    spatial_dims = 2
                    if 'ProbUnet' in base_cfg.model.framework:
                        base_cfg.model.framework = base_cfg.model.framework + '_old'
                        base_cfg.model.net['module_name'] =  base_cfg.model.net['module_name']+'_old'

                
                if spatial_dims == 3:
                    inference_mode = 'vol2vol'
                    print(f"Info: Spatial dimensions = 3. Using volumetric inference mode ({inference_mode}).")
                else:
                    inference_mode = 'vol2slice'
                    print(f"Info: Spatial dimensions = 2. Using slice-based inference mode ({inference_mode}).")
                
                # Check Max Projection compatibility with Volumetric models
                user_max_proj = use_max_proj_checkbox.value
                if inference_mode == 'vol2vol' and user_max_proj:
                    max_proj_setting = False
                else:
                    max_proj_setting = user_max_proj

                # Set configs
                base_cfg.mode = 'uncertainty_map'
                base_cfg.model.net['pred_slice2vol']['jupyter'] = True
                base_cfg.model.net['pred_slice2vol']['pixel_dim'] = pixel_dim
                base_cfg.model.net['pred_slice2vol']['multi_pred_mode'] = multi_pred_mode_dropdown.value
                base_cfg.model.net['pred_slice2vol']['n_samples'] = int(n_samples_text.value)
                
                # Inject dynamic settings
                base_cfg.model.net['pred_slice2vol']['inference_mode'] = inference_mode
                base_cfg.model.net['pred_slice2vol']['max_proj'] = max_proj_setting

                if '_old' in base_cfg.model.framework:
                    base_cfg.model.net['pred_slice2vol']['n_class_correction'] = base_cfg.model.net['params']['n_classes']
                else:
                    base_cfg.model.net['pred_slice2vol']['n_class_correction'] = base_cfg.model.net['params']['out_channels']
                
                base_cfg.model.net['pred_slice2vol']['remove_object_size'] = parse_int_list(remove_objects_text.value) if use_remove_objects_checkbox.value else None
                base_cfg.model.net['pred_slice2vol']['hole_size_threshold'] = parse_int_list(hole_size_threshold_text.value) if apply_holes_checkbox.value else None
                if apply_holes_checkbox.value: base_cfg.model.net['pred_slice2vol']['sv'] = True    
                base_cfg.model.net['pred_slice2vol']['min_thickness_list'] = parse_int_list(min_thickness_list_text.value) if apply_thickness_checkbox.value else None
                base_cfg.model.net['pred_slice2vol']['perycites_correction'] = apply_pericytes_checkbox.value
                
                base_cfg.data.inference_input.dir = Path(selected_path_base) / "split_3d"
                base_cfg.data.inference_input.data_type = ".tiff,.tif"
            except Exception as e:
                print(f"Config Error: {e}")
                return

            if mode == 'single model':
                if not ckpt_path_widget.selected:
                    print("Error: Select a CKPT file.")
                    return
                run_single_inference(copy.deepcopy(base_cfg), ckpt_path_widget.selected, Path(selected_path_base) / "model_predictions")
                print("######################## Prediction Ready #############################")

            elif mode == 'multi model':
                models_path = Path(models_folder_widget.selected)
                output_path = Path(output_path_widget.selected)
                structure = check_folder_structure() 
                inference_tasks = [] 

                if structure == "flat":
                    for f in models_path.glob("*.ckpt"):
                        inference_tasks.append((f, f"predictions_{f.stem}"))
                
                elif structure == "nested":
                    weight_opt = weight_options_dropdown.value
                    if weight_opt == "last":
                        for subdir in [x for x in models_path.iterdir() if x.is_dir()]:
                            ckpt = subdir / "checkpoints" / "last.ckpt"
                            if ckpt.exists(): inference_tasks.append((ckpt, f"predictions_{subdir.name}"))
                    elif weight_opt == "custom":
                        # Retrieve specific ckpt for each folder from the dynamic dropdowns
                        for widget in custom_weights_container.children:
                            if isinstance(widget, widgets.Dropdown):
                                ckpt_name = widget.value
                                folder = widget.target_folder
                                ckpt_path = models_path / folder / "checkpoints" / ckpt_name
                                if ckpt_path.exists():
                                    inference_tasks.append((ckpt_path, f"predictions_{folder}"))
                    elif weight_opt == "min":
                        for subdir in [x for x in models_path.iterdir() if x.is_dir()]:
                            ckpt_dir = subdir / "checkpoints"
                            if not ckpt_dir.exists(): continue
                            
                            best_ckpt = None
                            min_val_loss = float('inf')
                            
                            # Iterate over all ckpt files to find the one with lowest val_loss
                            for f in ckpt_dir.glob("*.ckpt"):
                                # Expecting format: epoch=int-val_loss=float.ckpt
                                if "val_loss=" in f.name:
                                    try:
                                        # Extract the number part after val_loss=
                                        loss_str = f.name.split("val_loss=")[1]
                                        # Remove .ckpt extension to parse the float
                                        loss_str = loss_str.replace(".ckpt", "")
                                        val_loss = float(loss_str)
                                        
                                        if val_loss < min_val_loss:
                                            min_val_loss = val_loss
                                            best_ckpt = f
                                    except ValueError:
                                        # Skip files that don't match the expected float format
                                        continue
                            
                            if best_ckpt:
                                inference_tasks.append((best_ckpt, f"predictions_{subdir.name}"))
                            else:
                                print(f"Warning: No valid 'val_loss' checkpoint found in {subdir.name}")

                if not inference_tasks:
                    print("No valid models found.")
                    return

                for i, (ckpt, out_name) in enumerate(inference_tasks):
                    output.clear_output(wait=True)
                    print(f"\nProcessing {i+1}/{len(inference_tasks)}: {out_name}")
                    run_single_inference(copy.deepcopy(base_cfg), ckpt, output_path / out_name)
                print(f"######################## Predictios for {len(inference_tasks)} models done #############################")
                print("\n######################## All Predictions Ready #############################")

    run_button.on_click(on_button_clicked)