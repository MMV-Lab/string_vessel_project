import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter

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
    out_path_pred = out_path_base / Path("pred")

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

    for fn in filenames:
        print(f"\nProcessing file: {fn.name}")
        reader = BioImage(fn)
        scene_list = reader.scenes
        print(f"scene_list: {scene_list}")

        # loop through all scenes
        for sname in scene_list:
            reader.set_scene(sname)
            # get image data
            img = reader.get_image_data("CZYX", T=0)

            # check if dimension_fix is necessary
            if use_dim_fix:
                if img.ndim < 4 or img.shape[0] == 0 or img.shape[1] == 0:
                    print(f"Warning: Image shape for scene {sname} in {fn.name} is not suitable for dimension_fix. Skipping dimension_fix.")
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

            # clean up filenames to get rid of spaces
            sname_cleaned = sname.replace("/", "_")

            # save individual multi-channel Tiff files
            out_fn = out_path_3d / f"{fn.stem}_{sname_cleaned}.tiff"
            OmeTiffWriter.save(im, out_fn, dim_order="CZYX")
            print(f"Saved: {out_fn}")

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

    get_channels_text_widget = widgets.Text(
        value='1, 2',
        description='Channels:',
        disabled=False,
        continuous_update=False
    )

    src_path_widget = widgets.Text(
        value='/path/to/string_vessel_data/',
        description='LIF files path:',
        disabled=False,
        continuous_update=False,
        layout=widgets.Layout(width='70%' , description_width='initial')
    )

    out_path_base_widget = widgets.Text(
        value='/path/to/output_folder/',
        description='Output path:',
        disabled=False,
        continuous_update=False,
        layout=widgets.Layout(width='70%' , description_width='initial')
    )

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
                get_channels_str = get_channels_text_widget.value
                src_path_str = src_path_widget.value
                out_path_base_str = out_path_base_widget.value

                try:
                    get_channels = [int(ch.strip()) for ch in get_channels_str.split(',') if ch.strip()]
                    if not get_channels:
                        raise ValueError("Channel list cannot be empty.")
                except ValueError:
                    print("Error: Channels must be comma-separated integers. E.g., '1, 2'")
                    return

                # Call the core processing logic
                _execute_processing_logic(use_dim_fix, get_channels, src_path_str, out_path_base_str)

            except Exception as e:
                print(f"An error occurred: {e}")

    run_button.on_click(on_run_button_clicked)

    # Display the widgets
    display(use_dim_fix_widget, get_channels_text_widget, src_path_widget, out_path_base_widget, run_button, output_area)
    