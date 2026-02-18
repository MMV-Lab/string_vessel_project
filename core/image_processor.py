import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from tqdm.notebook import tqdm
from ipyfilechooser import FileChooser
from bioio_base.types import PhysicalPixelSizes

def _execute_processing_logic(use_dim_fix, get_channels, src_path_str, out_path_base_str):
    """
    Core logic for processing LIF image files. This function is called internally.
    """
    print("Starting image processing...")

    # Convert paths to Path objects
    src = Path(src_path_str)
    out_path_base = Path(out_path_base_str)

    # specific where you want to save the splitted files
    out_path_3d = out_path_base / Path("split_3d")
    out_path_pred = out_path_base / Path("model_predictions")

    out_path_base.mkdir(parents=True, exist_ok=True)
    out_path_3d.mkdir(parents=True, exist_ok=True)
    out_path_pred.mkdir(parents=True, exist_ok=True)

    print(f"use_dim_fix: {use_dim_fix}")
    print(f"get_channels: {get_channels}")
    print(f"src: {src}")
    print(f"out_path_base: {out_path_base}")
    print(f"Output paths created: {out_path_base}, {out_path_3d}, {out_path_pred}")

    # get all LIF images
    filenames = sorted(list(src.glob("*.lif")))
    if not filenames:
        print(f"No .lif files found in: {src}")
        return

    print(f"Number of .lif files found: {len(filenames)}")

    for fn in tqdm(filenames, desc= "Split file progress", position=0):
    
        reader = BioImage(fn)
        scene_list = reader.scenes
    

        # loop through all scenes
        for sname in tqdm(scene_list, desc= "Split scene", leave=False, position=1):
            reader.set_scene(sname)
            # get image data
            img = reader.get_image_data("CZYX", T=0)

            # check if dimension_fix is necessary
            if use_dim_fix:
                if img.ndim < 4 or img.shape[0] == 0 or img.shape[1] == 0:
                    tqdm.write(f"Warning: Image shape for scene {sname} in {fn.name} is not suitable for dimension_fix. Skipping dimension_fix.")
                    im = img[get_channels, :, :, :]
                    
                else:
                    img_re = np.zeros_like(img)
                    counter_z = 0
                    counter_c = 0
                    for c in range(img.shape[0]):
                        for z in range(img.shape[1]):
                            if counter_c < img_re.shape[0] and counter_z < img_re.shape[1]:
                                img_re[counter_c, counter_z, :, :] = img[c, z, :, :]
                            counter_c += 1
                            if counter_c == img.shape[0]:
                                counter_c = 0
                                counter_z += 1
                    im = img_re[get_channels, :, :, :]
            else:
                im = img[get_channels, :, :, :]

            # clean up filenames and output path
            sname_cleaned = sname.replace("/", "_")
            out_fn = out_path_3d / f"{fn.stem}_{sname_cleaned}.tiff"

            # adding pixel info, normalize pps to a tuple (Z, Y, X)
            pps = getattr(reader, "physical_pixel_sizes", None)
            
            if pps is None:
                raw_voxel_sizes = (None, None, None)
            elif isinstance(pps, tuple):
                raw_voxel_sizes = pps  # tuple like (Z, Y, X)
            else:
                raw_voxel_sizes = (getattr(pps, "Z", None),
                                   getattr(pps, "Y", None),
                                   getattr(pps, "X", None))

            # Clean up missing/incorrect data, populate
            corrected_sizes = []
            axes_names = ["Z", "Y", "X"]
            
            for i, val in enumerate(raw_voxel_sizes):
                if val is None:
                    cleaned_val = 1.0
                else:
                    cleaned_val = float(val)
                    if cleaned_val < 0:
                        tqdm.write(f"  [Metadata Fix] Corrected negative pixel size for {axes_names[i]} in '{fn.name}' scene '{sname}': {cleaned_val} -> {abs(cleaned_val)}")
                        cleaned_val = abs(cleaned_val)
                corrected_sizes.append(cleaned_val)

            # Create corrected PhysicalPixelSizes object
            voxel_sizes = PhysicalPixelSizes(corrected_sizes[0], corrected_sizes[1], corrected_sizes[2])
     
            OmeTiffWriter.save(
                data=im,
                uri=out_fn,
                dim_order="CZYX",
                physical_pixel_sizes=voxel_sizes,
                physical_pixel_units="micron", 
            )
            
    print("\nImage processing completed!")


def run_image_processing_menu():
    """
    Displays the interactive widget menu for LIF image processing and handles execution.
    """
    print("Loading LIF Image Processing Configuration...")

    # Widgets for parameters
    use_dim_fix_widget = widgets.Checkbox(
        value=False,
        description='Use dimension fix?',
        disabled=False,
        indent=False
    )


    channel_header = widgets.Label(value="Channel specification")

    # Input for CD31 Channel (int1)
    cd31_widget = widgets.IntText(
        value=1,
        description='CD31 Channel:',
        disabled=False,
        layout=widgets.Layout(width='10%'),
        style={'description_width': '50%'} 
    )

    # Input for Col IV Channel (int2)
    col_iv_widget = widgets.IntText(
        value=2,
        description='Col IV Channel:',
        disabled=False,
        layout=widgets.Layout(width='10%'),
        style={'description_width': '50%'}
    )

    channels_container = widgets.HBox([cd31_widget, col_iv_widget])

    src_path_widget = FileChooser(
        Path.cwd().as_posix(), 
        title='Select LIF files folder',
        select_default=False 
    )
    src_path_widget.show_only_dirs = True 
    

    out_path_base_widget = FileChooser(
        Path.cwd().as_posix(), 
        title='Select Output Path',
        select_default=False 
    )
    out_path_base_widget.show_only_dirs = True

    run_button = widgets.Button(
        description='Run Processing',
        disabled=False,
        button_style='success',
        tooltip='Click to start processing LIF files',
        icon='play'
    )

    output_area = widgets.Output()

    def on_run_button_clicked(b):
        with output_area:
            clear_output()
            print("Initiating processing with current configuration...")

            try:
                use_dim_fix = use_dim_fix_widget.value
                src_path_str = src_path_widget.selected
                out_path_base_str = out_path_base_widget.selected

                if not src_path_str or not out_path_base_str:
                    raise ValueError("Please select and input/output file.")
                    return

       
                int1 = cd31_widget.value
                int2 = col_iv_widget.value
                if isinstance(int1,int) and isinstance(int2,int):
                    get_channels = [int1, int2]
                else:
                    raise ValueError("CD31 and Col VI channels should be integers.")
                    return

                # Call the core processing logic
                _execute_processing_logic(use_dim_fix, get_channels, src_path_str, out_path_base_str)

            except Exception as e:
                print(f"An error occurred: {e}")

    run_button.on_click(on_run_button_clicked)

    # Display the widgets
    display(
        use_dim_fix_widget, 
        channel_header,     
        channels_container, 
        src_path_widget, 
        out_path_base_widget, 
        run_button, 
        output_area
    )
    
