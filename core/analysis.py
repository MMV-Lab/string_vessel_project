from vessel_analysis_3d.processing_pipeline import Pipeline3D
from skimage.morphology import skeletonize
import numpy as np
from pathlib import Path
import pandas as pd
from bioio import BioImage
from skimage.morphology import label, remove_small_objects, remove_small_holes
from vessel_analysis_3d.graph.networkx_from_array import get_networkx_graph_from_array
from vessel_analysis_3d.graph.core import GraphObj
from vessel_analysis_3d.graph.stats_reporting import report_everything

import ipywidgets as widgets
from IPython.display import display, clear_output
from tqdm import tqdm

def analysis_menu():
    """
    Provides an interactive menu with editable text fields to set parameters
    and run the vessel analysis pipeline.
    """
    # Default parameters
    default_params = {
        "pixelDimensions": "1.0, 1.0, 1.0", # Stored as string for Text widget
        "pruningScale": 1,
        "lengthLimit": 1,
        "diaScale": 1,
        "branchingThreshold": 0.25
    }
    
    
    # Create widgets for each parameter
    pixel_dim_widget = widgets.Text(
        value=default_params["pixelDimensions"],
        description='Pixel Dimensions:',
        layout=widgets.Layout(description_width='initial'),
        style={'description_width': '40%'}
    )
    pruning_scale_widget = widgets.FloatText(
        value=default_params["pruningScale"],
        description='Pruning Scale:',
        layout=widgets.Layout( description_width='initial'),
        style={'description_width': '30%'}
    )
    length_limit_widget = widgets.FloatText(
        value=default_params["lengthLimit"],
        description='Length Limit:',
        layout=widgets.Layout( description_width='initial')
    )
    dia_scale_widget = widgets.FloatText(
        value=default_params["diaScale"],
        description='Dia Scale:',
        layout=widgets.Layout( description_width='initial')
    )
    branching_threshold_widget = widgets.FloatText(
        value=default_params["branchingThreshold"],
        description='Branching Threshold:',
        layout=widgets.Layout( description_width='initial'),
        style={'description_width': '50%'}
    )

    # Path widgets
    default_pred_path = "/path/to/segmentation/results"
    default_out_path = "/path/to/save/all/stats"

    pred_path_widget = widgets.Text(
        value=default_pred_path,
        description='Input Path:',
        layout=widgets.Layout( width='70%' , description_width='initial')
    )
    out_path_widget = widgets.Text(
        value=default_out_path,
        description='Output Path:',
        layout=widgets.Layout( width='70%', description_width='initial')
    )

    run_button = widgets.Button(
        description="Run Analysis",
        button_style='success', 
    )
    output_area = widgets.Output()

    def on_button_click(b):
        with output_area:
            clear_output()
            print("Starting analysis with the following parameters:")

            # Construct params dictionary from widget values
            params = {}
            try:
                params["pixelDimensions"] = [float(d.strip()) for d in pixel_dim_widget.value.split(',')]
            except ValueError:
                print("Invalid input for Pixel Dimensions. Please use a comma-separated list of numbers.")
                return

            params["pruningScale"] = pruning_scale_widget.value
            params["lengthLimit"] = length_limit_widget.value
            params["diaScale"] = dia_scale_widget.value
            params["branchingThreshold"] = branching_threshold_widget.value

            for key, value in params.items():
                print(f"  {key}: {value}")

            pred_path = Path(pred_path_widget.value)
            out_path = Path(out_path_widget.value)

            print(f"  Segmentation Results Path: {pred_path}")
            print(f"  Save Statistics Path: {out_path}")

            # Ensure output path exists
            out_path.mkdir(parents=True, exist_ok=True)

            filenames = sorted(pred_path.glob("*.tiff"))
            if not filenames:
                print(f"No .tiff files found in {pred_path}. Please check the path and file extensions.")
                return

            save_name = out_path / Path("data_full_stats.csv")

            stats = []
            print(f"{len(filenames)} files found to be analized")
            for fn in tqdm(filenames, desc= "Analysis file progress "):
               
                

                fileID = fn.stem
                pred = BioImage(fn).get_image_data("ZYX", C=0, T=0)

                # here, in this demo, we run analysis on string vessel only and on all vessels
                # we can also run on normal vessel only (pred == 1)
                string_vessel = pred == 2
                all_vessel = pred > 0

                _, num_string = label(string_vessel, connectivity=3, return_num=True)

                all_skl = skeletonize(all_vessel > 0, method="lee").astype(np.uint8)
                all_skl[all_skl > 0] = 1
                _, _, _, all_reports = Pipeline3D.process_one_file(all_vessel, all_skl, params)

                raw_stats_name = out_path / f"{fileID}_all_vessels_stats.csv"
                all_reports[0].to_csv(raw_stats_name, index=False)

                if num_string > 0:

                    string_skl = skeletonize(string_vessel > 0, method="lee").astype(np.uint8)
                    string_skl[string_skl > 0] = 1

                    # skeleton to graph
                    networkxGraph = get_networkx_graph_from_array(string_skl)

                    # Statistical Analysis
                    gh = GraphObj(string_vessel, string_skl, networkxGraph, **params)
                    skl_final = gh.prune_and_analyze(return_final_skel=True)

                    if np.count_nonzero(skl_final) < 3:
                        stats.append(
                            {
                                "filename": fileID,
                                "num_string_vessel": 0,
                                "average_thickness_string": 0,
                                "average_straightness_string": 0,
                                "average_length_string": 0,
                                "string_to_all_ratio": 0,
                                "average_thickness_all": np.mean(all_reports[0]["diameter"]),
                                "average_straightness_all": np.mean(all_reports[0]["straightness"]),
                                "average_length_all": np.mean(all_reports[0]["length"]),
                            }
                        )
                    else:
                        string_reports = report_everything(gh, "default")
                        # save the string report
                        raw_string_stats_name = out_path / f"{fileID}_string_vessels_stats.csv"
                        string_reports[0].to_csv(raw_string_stats_name, index=False)
                        stats.append(
                            {
                                "filename": fileID,
                                "num_string_vessel": num_string,
                                "average_thickness_string": np.mean(string_reports[0]["diameter"]),
                                "average_straightness_string": np.mean(string_reports[0]["straightness"]),
                                "average_length_string": np.mean(string_reports[0]["length"]),
                                "string_to_all_ratio": len(string_reports[0]) / len(all_reports[0]),
                                "average_thickness_all": np.mean(all_reports[0]["diameter"]),
                                "average_straightness_all": np.mean(all_reports[0]["straightness"]),
                                "average_length_all": np.mean(all_reports[0]["length"]),
                            }
                        )
                else:
                    stats.append(
                        {
                            "filename": fileID,
                            "num_string_vessel": 0,
                            "average_thickness_string": 0,
                            "average_straightness_string": 0,
                            "average_length_string": 0,
                            "string_to_all_ratio": 0,
                            "average_thickness_all": np.mean(all_reports[0]["diameter"]),
                            "average_straightness_all": np.mean(all_reports[0]["straightness"]),
                            "average_length_all": np.mean(all_reports[0]["length"]),
                        }
                    )

            stats_df = pd.DataFrame(stats)
            stats_df.to_csv(save_name, index=False)
            print(f"\nAnalysis complete. Summary saved to: {save_name}")
            print(f"Individual raw statistics saved to: {out_path}")


    run_button.on_click(on_button_click)

    # Display the widgets
    display(
        widgets.VBox([
            widgets.HTML("<b>Vessel Analysis Parameters</b>"),
            pixel_dim_widget,
            pruning_scale_widget,
            length_limit_widget,
            dia_scale_widget,
            branching_threshold_widget,
            widgets.HTML("<b>Input/Output Paths</b>"),
            pred_path_widget,
            out_path_widget,
            run_button,
            output_area
        ])
    )
    
