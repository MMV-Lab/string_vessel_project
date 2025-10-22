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
from tqdm.notebook import tqdm
import gc
import torch
import bioio_tifffile


logger = getLogger(__name__)

T = TypeVar("T")


def Remove_objects(seg_full, n_classes, remove_object_size):
    """
    Applies removal of small objects to all object classes (1 to n_classes-1) 
    in a 3D segmentation volume, allowing for class-specific size thresholds.

    Args:
        seg_full (np.ndarray): The 3D segmentation volume with integer class values.
        n_classes (int): The total number of classes in the segmentation (including background).
        remove_object_size (list or int): A single minimum size (int) or a list of 
                                          minimum sizes (list) for objects to be kept. 
                                          If a list, its length must be 1 or equal 
                                          to the number of object classes (n_classes - 1).

    Returns:
        np.ndarray: The segmentation volume with small objects removed for each class.

    Raises:
        ValueError: If the length of remove_object_size list is invalid.
    """
    
    classes_to_process = range(1, n_classes)
    num_target_classes = len(classes_to_process)
    
    thresholds = []

    if not isinstance(remove_object_size, list):
        thresholds = [remove_object_size] * num_target_classes
    else:
        list_len = len(remove_object_size)
        
        if list_len == 1:
            thresholds = [remove_object_size[0]] * num_target_classes
        
        elif list_len == num_target_classes:
            thresholds = remove_object_size
            
        else: 
            raise ValueError(
                f"The list 'remove_object_size' has {list_len} elements, "
                f"but {num_target_classes} (or 1) were expected for the {num_target_classes} classes to process "
                f"(Class 1 to {n_classes - 1}). The background (Class 0) is ignored."
            )
            
    seg_cleaned = np.zeros_like(seg_full)
    
    for i, class_id in enumerate(classes_to_process):
        min_size_threshold = thresholds[i]
        
        seg_class_mask = (seg_full == class_id)
        
        if seg_class_mask.any():
            seg_class_clean = remove_small_objects(seg_class_mask, min_size=min_size_threshold)
            seg_cleaned[seg_class_clean] = class_id
            
    return seg_cleaned

def Hole_Correction(seg_full, n_classes, hole_size_threshold):
    """
    Applies hole correction to multiple classes in a segmentation volume.

    The correction is applied to object classes (typically 1 up to n_classes-1).
    Each class can have a different hole size threshold. It also includes 
    an initial removal of small objects for all classes.

    Args:
        seg_full (np.ndarray): The 3D segmentation volume with integer class values.
        n_classes (int): The total number of classes in the segmentation (including background).
        hole_size_threshold (list or int): A single threshold (int) or a list of 
                                           thresholds (list) for hole correction. 
                                           If a list, its length must be 1 or 
                                           equal to the number of object classes (n_classes - 1).

    Returns:
        np.ndarray: The corrected segmentation volume.

    Raises:
        ValueError: If the length of hole_size_threshold is not 1 and is less than n_classes - 1.
    """

    classes_to_correct = range(1, n_classes)
    num_target_classes = len(classes_to_correct)
    
    thresholds = []

    if not isinstance(hole_size_threshold, list):
        thresholds = [hole_size_threshold] * num_target_classes
    else:
        list_len = len(hole_size_threshold)
        
        if list_len == 1:
            thresholds = [hole_size_threshold[0]] * num_target_classes
        
        elif list_len == num_target_classes:
            thresholds = hole_size_threshold
            
        else: 
            raise ValueError(
                f"The list 'hole_size_threshold' has {list_len} elements, "
                f"but {num_target_classes} (or 1) were expected for the {num_target_classes} classes to correct "
                f"(Class 1 to {n_classes - 1}). The background (Class 0) does not need a threshold."
            )
            
    seg_corrected = seg_full.copy()
    
    for i, class_id in enumerate(classes_to_correct):
        threshold = thresholds[i]
        seg_obj_mask = (seg_corrected == class_id)
        
        if seg_obj_mask.any():
            seg_obj_slice_corrected = seg_obj_mask.copy() 
            
            for zz in range(seg_full.shape[0]):
                s_v = remove_small_holes(seg_obj_slice_corrected[zz, :, :], area_threshold=threshold)
                seg_obj_slice_corrected[zz, :, :] = s_v[:, :]   
            seg_corrected[seg_corrected == class_id] = 0 
            seg_corrected[seg_obj_slice_corrected] = class_id 
            
    return seg_corrected
    
def Thickness_Corretion(seg_full, n_classes, min_thickness_list):
    """
    Applies topology-preserving thinning (thickness correction) to all object 
    classes (1 to n_classes-1) in a 3D segmentation volume, using a specific
    minimum thickness for each class. Class 0 (background) is automatically ignored.

    Args:
        seg_full (np.ndarray): The 3D segmentation volume with integer class values.
        n_classes (int): The total number of classes in the segmentation (including Class 0).
        min_thickness_list (list or np.ndarray): A list or array of minimum
                                                thickness values. The index 'i' 
                                                corresponds to the minimum thickness for Class i+1.
                                                (e.g., index 0 is for Class 1).

    Returns:
        np.ndarray: The segmentation volume where each object class has been thinned, 
                    preserving its original class label.
    """
    

    classes_to_process = range(1, n_classes)
    num_object_classes = len(classes_to_process)
    
    # 1. Validate the length of the minimum thickness list
    if len(min_thickness_list) != num_object_classes:
        raise ValueError(
            f"The length of 'min_thickness_list' ({len(min_thickness_list)}) does not match "
            f"the number of object classes to process ({num_object_classes}). "
            "Class 0 (background) is ignored, so {num_object_classes} values are expected "
            " (one for each class from 1 to {n_classes-1})."
        )


    seg_corrected = np.zeros_like(seg_full)

    for i, class_id in enumerate(classes_to_process):
        
        current_min_thickness = min_thickness_list[i]
        seg_class_mask = (seg_full == class_id)
        
        seg_thinned = topology_preserving_thinning(
            seg_class_mask, 
            min_thickness=current_min_thickness,  
            thin=1
        )
    
        seg_corrected[seg_thinned > 0] = class_id
        
    return seg_corrected


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

    # --- Inference Options --- section
    inference_header = widgets.Label(
        value="--- Inference Options ---",
        style={'font_weight': 'bold'}
    )

    # Checkbox for Use Max projection
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

    # Checkboxes for post-processing
    apply_pericytes_checkbox = widgets.Checkbox(
        value=True,
        description="Apply Pericytes Correction",
        disabled=False
    )

    apply_thickness_checkbox = widgets.Checkbox(
        value=True,
        description="Apply Thickness Adjustment",
        disabled=False
    )
    min_thickness_list_text = widgets.Text(
        value="", 
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
        value="", 
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
        value="", # Default value for the original implementation
        placeholder="e.g., 15, 30",
        description="Hole Thresholds :",
        disabled=False,
        layout=widgets.Layout(width='80%')
    )

    def toggle_thickness_text(change):
        min_thickness_list_text.layout.display = 'block' if change['new'] else 'none'

    def toggle_remove_objects_text(change):
        remove_objects_text.layout.display = 'block' if change['new'] else 'none'

    def toggle_hole_threshold_text(change):
        hole_size_threshold_text.layout.display = 'block' if change['new'] else 'none'

    min_thickness_list_text.layout.display = 'block' if apply_thickness_checkbox.value else 'none'
    remove_objects_text.layout.display = 'block' if use_remove_objects_checkbox.value else 'none'
    hole_size_threshold_text.layout.display = 'block' if apply_holes_checkbox.value else 'none' 

    apply_thickness_checkbox.observe(toggle_thickness_text, names='value')
    use_remove_objects_checkbox.observe(toggle_remove_objects_text, names='value')
    apply_holes_checkbox.observe(toggle_hole_threshold_text, names='value')

    run_button = widgets.Button(description="Run Inference" , button_style='success', )
    output = widgets.Output()

    display(yaml_path_text, ckpt_path_text, path_base_text, 
                inference_header, use_max_proj_checkbox,
                postprocessing_header, 
                use_remove_objects_checkbox, remove_objects_text,
                apply_holes_checkbox, hole_size_threshold_text,
                apply_thickness_checkbox, min_thickness_list_text, 
                apply_pericytes_checkbox, 
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
            selected_yaml = yaml_path_text.value
            selected_ckpt = ckpt_path_text.value
            selected_path_base = path_base_text.value
            
            apply_max_proj = use_max_proj_checkbox.value
            apply_pericytes = apply_pericytes_checkbox.value
            apply_thickness = apply_thickness_checkbox.value
            apply_remove_objects = use_remove_objects_checkbox.value
            apply_holes = apply_holes_checkbox.value

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
                out_p = path_base / Path("model_predictions")
                out_p.mkdir(parents=True, exist_ok=True)

                filenames = sorted(input_path.glob("*.tiff"))
                filenames.extend(list(input_path.glob('*.tif')))

                print(f"{len(filenames)} files found to predict")
                for fn in tqdm(filenames, desc= "Prediction file progress", position=0):
                    try:
                        img = BioImage(fn, reader=bioio_tifffile.Reader).get_image_data("CZYX", T=0)
                    except Exception as e:
                        try:
                            img = BioImage(fn).get_image_data("CZYX", T=0)
                        except Exception as e:
                            raise ValueError("Error at reading time.")
                    
                    if apply_max_proj:
                        img = np.max(img, axis=1)
                        img = np.expand_dims(img, axis=1)


                    out_list = []    
                    for zz in range(img.shape[1]):
                        im_input = img[:, zz, :, :]
                        seg = executor.process_one_image(im_input)
                        seg_np = np.squeeze(seg)
                        gc.collect()
                        torch.cuda.empty_cache()
                        out_list.append(seg_np)
                    seg_full = np.stack(out_list, axis=0)


                    if apply_pericytes:
                       
                        seg_2 = remove_small_objects(seg_full == 2, min_size=30)
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

                    if  apply_remove_objects:
                        seg_full = Remove_objects(seg_full=seg_full, n_classes=3, remove_object_size=remove_objects_sizes)
     
                    if apply_holes:
                        seg_full = Hole_Correction(seg_full=seg_full, n_classes=3, hole_size_threshold=hole_size_thresholds)   

                    # Thickness adjustment
                    if apply_thickness:
                       seg_full = Thickness_Corretion(seg_full=seg_full, n_classes=3,min_thickness_list= min_thickness_list_values)

                    tqdm.write(" " * 40, end="\r")
                    if 'ome.tiff' in fn.name:
                        out_fn = out_p / fn.name.replace('ome.tiff','_pred.tiff')
                    elif 'ome.tif' in fn.name:
                        out_fn = out_p / fn.name.replace('ome.tif','_pred.tif')
                    elif '.tiff' in fn.name:
                        out_fn = out_p / fn.name.replace('.tiff','_pred.tiff')
                    elif '.tif' in fn.name:
                        out_fn = out_p / fn.name.replace('.tif','_pred.tif')

                    OmeTiffWriter.save(seg_full, out_fn, dim_order="ZYX")
                print("Inference completed.")

            except Exception as e:
                print(f"An error occurred: {e}")

    run_button.on_click(on_button_clicked)
