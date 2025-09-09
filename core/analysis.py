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
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
from tqdm.notebook import tqdm
import networkx as nx

def calculate_orientation_and_direction(graph, pixel_dims):
    """
    Calculates the inclination angle and direction for each segment of the graph.
    """
    segment_stats = []
    
    # Reference axes for angle calculation
    axis_vectors = {
        'Z_angle': np.array([1, 0, 0]),
        'Y_angle': np.array([0, 1, 0]),
        'X_angle': np.array([0, 0, 1])
    }
    
    # Scale factors from pixel dimensions
    scale_factors = np.array(pixel_dims)
    
    for u, v, data in graph.edges(data=True):
        if 'voxel_coords' in data:
            segment_coords = np.array(data['voxel_coords'])
            if len(segment_coords) > 1:
                # The segment's direction vector is the difference between the last and first point
                direction_vector = segment_coords[-1] - segment_coords[0]
                
                # Apply pixel dimensions to the direction vector
                scaled_direction_vector = direction_vector * scale_factors
                
                # Normalize the scaled direction vector for angle calculation
                norm_direction = scaled_direction_vector / np.linalg.norm(scaled_direction_vector)
                
                # Convert coordinates to a list of standard Python integers for a cleaner string representation
                start_coords = [int(coord) for coord in segment_coords[0]]
                end_coords = [int(coord) for coord in segment_coords[-1]]
                
                stats_entry = {'segment_id': f'{tuple(start_coords)}-{tuple(end_coords)}'}
                
                # Calculate the angle with each axis
                for axis_name, axis_vec in axis_vectors.items():
                    # Use the dot product to find the cosine of the angle
                    dot_product = np.dot(norm_direction, axis_vec)
                    # Arccosine to get the angle in radians, then convert to degrees
                    angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
                    stats_entry[axis_name] = angle_deg
                
                segment_stats.append(stats_entry)
                
    return pd.DataFrame(segment_stats)

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
            print(f"{len(filenames)} files found to be analyzed")
            for fn in tqdm(filenames, desc= "Analysis file progress "):
                
                try:
                    fileID = fn.stem
                    pred = BioImage(fn).get_image_data("ZYX", C=0, T=0)
                    
                    # here, in this demo, we run analysis on string vessel only and on all vessels
                    string_vessel = pred == 2
                    all_vessel = pred > 0

                    _, num_string = label(string_vessel, connectivity=3, return_num=True)

                    all_skl = skeletonize(all_vessel > 0, method="lee").astype(np.uint8)
                    all_skl[all_skl > 0] = 1
                    
                    # Create the all_vessel graph and add voxel_coords
                    networkxGraph_all = get_networkx_graph_from_array(all_skl)
                    for u, v in networkxGraph_all.edges():
                        networkxGraph_all[u][v]['voxel_coords'] = [u, v]
                        
                    _, _, _, all_reports = Pipeline3D.process_one_file(all_vessel, all_skl, params)

                    raw_stats_name = out_path / f"{fileID}_all_vessels_stats.csv"
                    all_reports[0].to_csv(raw_stats_name, index=False)

                    if num_string > 0:
                        string_skl = skeletonize(string_vessel > 0, method="lee").astype(np.uint8)
                        string_skl[string_skl > 0] = 1

                        # skeleton to graph
                        networkxGraph_string = get_networkx_graph_from_array(string_skl)
                        
                        # Add 'voxel_coords' to the string vessel graph edges
                        for u, v in networkxGraph_string.edges():
                            networkxGraph_string[u][v]['voxel_coords'] = [u, v]
                        
                        # orientation calc
                        orientation_df = calculate_orientation_and_direction(networkxGraph_string, params["pixelDimensions"])
                        
                        # Save orientation stats
                        orientation_stats_name = out_path / f"{fileID}_string_vessels_orientation.csv"
                        orientation_df.to_csv(orientation_stats_name, index=False)
                        
                        # Statistical Analysis
                        gh = GraphObj(string_vessel, string_skl, networkxGraph_string, **params)
                        skl_final = gh.prune_and_analyze(return_final_skel=True)

                        if np.count_nonzero(skl_final) < 3:
                            stats.append(
                                {
                                    "filename": fileID,
                                    "num_string_vessel": 0,
                                    "average_thickness_string": 0,
                                    "average_straightness_string": 0,
                                    "average_length_string": 0,
                                    "sum_length_string": 0,
                                    "string_to_all_ratio": 0,
                                    "average_thickness_all": np.mean(all_reports[0]["diameter"]),
                                    "average_straightness_all": np.mean(all_reports[0]["straightness"]),
                                    "average_length_all": np.mean(all_reports[0]["length"]),
                                    "sum_length_all": np.sum(all_reports[0]["length"]),
                                    "sum_length_string_to_all_ratio": 0,
                                    "average_Z_angle_string": 0, 
                                    "average_Y_angle_string": 0,
                                    "average_X_angle_string": 0,
                                }
                            )
                        else:
                            string_reports = report_everything(gh, "default")
                            raw_string_stats_name = out_path / f"{fileID}_string_vessels_stats.csv"
                            string_reports[0].to_csv(raw_string_stats_name, index=False)
                            stats.append(
                                {
                                    "filename": fileID,
                                    "num_string_vessel": num_string,
                                    "average_thickness_string": np.mean(string_reports[0]["diameter"]),
                                    "average_straightness_string": np.mean(string_reports[0]["straightness"]),
                                    "average_length_string": np.mean(string_reports[0]["length"]),
                                    "sum_length_string": np.sum(string_reports[0]["length"]),
                                    "string_to_all_ratio": len(string_reports[0]) / len(all_reports[0]),
                                    "average_thickness_all": np.mean(all_reports[0]["diameter"]),
                                    "average_straightness_all": np.mean(all_reports[0]["straightness"]),
                                    "average_length_all": np.mean(all_reports[0]["length"]),
                                    "sum_length_all": np.sum(all_reports[0]["length"]),
                                    "sum_length_string_to_all_ratio": np.sum(string_reports[0]["length"]) / np.sum(all_reports[0]["length"]),
                                    "average_Z_angle_string": np.mean(orientation_df["Z_angle"]), 
                                    "average_Y_angle_string": np.mean(orientation_df["Y_angle"]),
                                    "average_X_angle_string": np.mean(orientation_df["X_angle"]),
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
                                "sum_length_string": 0,
                                "string_to_all_ratio": 0,
                                "average_thickness_all": np.mean(all_reports[0]["diameter"]),
                                "average_straightness_all": np.mean(all_reports[0]["straightness"]),
                                "average_length_all": np.mean(all_reports[0]["length"]),
                                "sum_length_all": np.sum(all_reports[0]["length"]),
                                "sum_length_string_to_all_ratio": 0,
                                "average_Z_angle_string": 0, 
                                "average_Y_angle_string": 0,
                                "average_X_angle_string": 0,
                            }
                        )
                except Exception as e:
                    with open(out_path / Path("Unprocessed_files.txt"), "a") as f:
                        f.write(f"{fn.name}\n")
                    

            stats_df = pd.DataFrame(stats)
            stats_df.to_csv(save_name, index=False)
            print(f"\nAnalysis complete. Summary saved to: {save_name}")
            print(f"Individual raw statistics saved to: {out_path}")
            if os.path.exists(out_path / Path("Unprocessed_files.txt")):
                print("#################################Note#################################")
                print("Some files could not be processed due to format or content errors")
                print("The name of the unprocessed files are saved in:")
                print(out_path / Path("Unprocessed_files.txt"))  

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