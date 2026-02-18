from vessel_analysis_3d.processing_pipeline import Pipeline3D
from skimage.morphology import skeletonize, remove_small_objects
from bioio import BioImage
import bioio_tifffile
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipyfilechooser import FileChooser
from tqdm.notebook import tqdm
from skan import Skeleton, summarize
import matplotlib.pyplot as plt


def analyze_skeleton_skan(skeleton_image, pixel_dims):
    """Uses skan to efficiently calculate orientation, topology, and advanced PCA features."""
    if np.count_nonzero(skeleton_image) == 0:
        return pd.DataFrame(columns=["Z_angle", "Y_angle", "X_angle"]), {
            "branch_points": 0, "endpoints": 0, 
            "cyclomatic_number": 0, "avg_junction_valence": 0,
            "anisotropy_std": 0
        }

    skel = Skeleton(skeleton_image, spacing=pixel_dims)
    branch_data = summarize(skel,separator='-')

    # --- Orientation Calculation ---
    coords_src = branch_data[['image-coord-src-0', 'image-coord-src-1', 'image-coord-src-2']].to_numpy()
    coords_dst = branch_data[['image-coord-dst-0', 'image-coord-dst-1', 'image-coord-dst-2']].to_numpy()
    
    vectors = coords_dst - coords_src
    norms = np.linalg.norm(vectors, axis=1)
    
    valid_mask = norms > 0
    vectors = vectors[valid_mask]
    normalized_vectors = vectors / norms[valid_mask, np.newaxis]

    axes = {'Z_angle': np.array([1, 0, 0]), 'Y_angle': np.array([0, 1, 0]), 'X_angle': np.array([0, 0, 1])}
    orientation_results = {}
    for axis_name, axis_vec in axes.items():
        dots = np.abs(np.dot(normalized_vectors, axis_vec)) 
        angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        orientation_results[axis_name] = angles

    orientation_df = pd.DataFrame(orientation_results)

    # --- Topology Statistics ---
    degrees = skel.degrees
    branch_points = np.sum(degrees > 2)
    endpoints = np.sum(degrees == 1)
    
    n_edges = branch_data.shape[0]
    n_nodes = np.sum(degrees != 2) 
    cyclomatic_number = max(0, n_edges - n_nodes + 1)

    junction_degrees = degrees[degrees > 2]
    avg_junction_valence = np.mean(junction_degrees) if len(junction_degrees) > 0 else 0

    if not orientation_df.empty:
        anisotropy_std = np.mean([np.std(orientation_results['Z_angle']), 
                                  np.std(orientation_results['Y_angle']), 
                                  np.std(orientation_results['X_angle'])])
    else:
        anisotropy_std = 0

    topo_stats = {
        "branch_points": branch_points,
        "endpoints": endpoints,
        "cyclomatic_number": cyclomatic_number,
        "avg_junction_valence": avg_junction_valence,
        "anisotropy_std": anisotropy_std
    }

    return orientation_df, topo_stats

def compile_statistics_row(file_id, volume, params, data_sources):
    """Consolidates statistics matching old columns + new PCA features."""
    
    src_all = data_sources.get('all', (None, {}, None))
    src_string = data_sources.get('string', (None, {}, None))
    src_normal = data_sources.get('normal', (None, {}, None))

    def get_basics(source_tuple):
        df, topo, orient = source_tuple
        if df is None or df.empty:
            return {k: 0 for k in ["num", "avg_dia", "std_dia", "avg_str", "std_str", "avg_len", "std_len", "sum_len", "Z", "Y", "X"]}
        
        return {
            "num": len(df),
            "avg_dia": np.mean(df["diameter"]),
            "std_dia": np.std(df["diameter"]),
            "avg_str": np.mean(df["straightness"]),
            "std_str": np.std(df["straightness"]),
            "avg_len": np.mean(df["length"]),
            "std_len": np.std(df["length"]),
            "sum_len": np.sum(df["length"]),
            "Z": np.mean(orient["Z_angle"]) if (orient is not None and not orient.empty) else 0,
            "Y": np.mean(orient["Y_angle"]) if (orient is not None and not orient.empty) else 0,
            "X": np.mean(orient["X_angle"]) if (orient is not None and not orient.empty) else 0,
        }

    s_all = get_basics(src_all)
    s_string = get_basics(src_string)
    s_normal = get_basics(src_normal)

    t_all = src_all[1] if src_all[1] else {"branch_points": 0, "endpoints": 0}
    t_string = src_string[1] if src_string[1] else {"branch_points": 0, "endpoints": 0}
    t_normal = src_normal[1] if src_normal[1] else {"branch_points": 0, "endpoints": 0}

    count_all = s_all["num"] if s_all["num"] > 0 else 1
    len_all = s_all["sum_len"] if s_all["sum_len"] > 0 else 1
    string_to_all_ratio = s_string["num"] / count_all if s_all["num"] > 0 else 0
    sum_length_string_ratio = s_string["sum_len"] / len_all if s_all["sum_len"] > 0 else 0

    # --- ROW CONSTRUCTION ---
    row = {
        "filename": file_id,

        # --- STRING VESSELS ---
        "num_string_vessel": s_string["num"],
        "num_string_density": s_string["num"] / volume if volume > 0 else 0,
        "avg_thickness_string": s_string["avg_dia"],
        "std_thickness_string": s_string["std_dia"],
        "avg_straightness_string": s_string["avg_str"],
        "std_straightness_string": s_string["std_str"],
        "average_length_string": s_string["avg_len"],
        "std_average_length_string": s_string["std_len"],
        "sum_length_string": s_string["sum_len"],
        "string_to_all_ratio": string_to_all_ratio,
        "sum_length_string_to_all_ratio": sum_length_string_ratio,

        # --- NORMAL VESSELS ---
        "avg_thickness_normal": s_normal["avg_dia"],
        "std_thickness_normal": s_normal["std_dia"],
        "avg_straightness_normal": s_normal["avg_str"],
        "std_straightness_normal": s_normal["std_str"],
        "average_length_normal": s_normal["avg_len"],
        "std_length_normal": s_normal["std_len"],
        "sum_length_normal": s_normal["sum_len"],
        "length_density_normal": s_normal["sum_len"] / volume,
        "branch_point_density_normal": t_normal.get("branch_points", 0) / volume,
        "endpoint_density_normal": t_normal.get("endpoints", 0) / volume,

        # --- ALL VESSELS ---
        "avg_thickness_all": s_all["avg_dia"],
        "std_thickness_all": s_all["std_dia"],
        "avg_straightness_all": s_all["avg_str"],
        "std_straightness_all": s_all["std_str"],
        "average_length_all": s_all["avg_len"],
        "std_length_all": s_all["std_len"],
        "sum_length_all": s_all["sum_len"],
        "length_density_all": s_all["sum_len"] / volume,
        "branch_point_density_all": t_all.get("branch_points", 0) / volume,
        "endpoint_density_all": t_all.get("endpoints", 0) / volume,

        # --- ORIENTATION ---
        "average_Z_angle_string": s_string["Z"], "average_Y_angle_string": s_string["Y"], "average_X_angle_string": s_string["X"],
        "average_Z_angle_normal": s_normal["Z"], "average_Y_angle_normal": s_normal["Y"], "average_X_angle_normal": s_normal["X"],

        # --- METADATA ---
        "pixel_size_Z": params["pixelDimensions"][0],
        "pixel_size_Y": params["pixelDimensions"][1],
        "pixel_size_X": params["pixelDimensions"][2],
        "num_slices": params.get("num_slices", 0),
        "height": params.get("height", 0),
        "width": params.get("width", 0),
        "volume": volume,
    }

    # --- NEW FEATURES (PCA) ---
    row["cyclomatic_index_all"] = t_all.get("cyclomatic_number", 0)
    row["cyclomatic_index_normal"] = t_normal.get("cyclomatic_number", 0)
    row["cyclomatic_index_string"] = t_string.get("cyclomatic_number", 0)
    
    row["junction_valence_all"] = t_all.get("avg_junction_valence", 0)
    row["junction_valence_normal"] = t_normal.get("avg_junction_valence", 0)
    row["junction_valence_string"] = t_string.get("avg_junction_valence", 0)
    
    row["anisotropy_std_all"] = t_all.get("anisotropy_std", 0)
    row["anisotropy_std_normal"] = t_normal.get("anisotropy_std", 0)
    row["anisotropy_std_string"] = t_string.get("anisotropy_std", 0)

    row["num_all_density"] = s_all["num"] / volume if volume > 0 else 0
    row["num_normal_density"] = s_normal["num"] / volume if volume > 0 else 0

    return row

def process_vessel_type(vessel_mask, params, fileID, output_root, label_type):
    """
    Standardized processing for any vessel type.
    Includes FIX for removing metadata keys and checking report list validity.
    """
    if np.count_nonzero(vessel_mask) == 0:
        return None, None, None

    skl = skeletonize(vessel_mask, method="lee").astype(np.uint8)
    skl = remove_small_objects(skl.astype(bool), max_size=4).astype(np.uint8)
    skl[skl > 0] = 1

    if np.count_nonzero(skl) == 0:
        return None, None, None

    # Skan Analysis
    orient_df, topo_stats = analyze_skeleton_skan(skl, params["pixelDimensions"])
    
    # Pipeline3D Analysis
    # Filter params to exclude metadata keys that confuse GraphObj
    forbidden = ['num_slices', 'height', 'width']
    clean_params = {k: v for k, v in params.items() if k not in forbidden}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # reports is typically a LIST of DataFrames or tuples
        _, _, _, reports = Pipeline3D.process_one_file(vessel_mask, skl, clean_params)

    final_report_df = None
    if reports and len(reports) > 0:
        final_report_df = reports[0]
        final_report_df.to_csv(output_root / f"{fileID}_{label_type}_vessels_stats.csv", index=False)
    
    if orient_df is not None and not orient_df.empty:
        orient_df.to_csv(output_root / f"{fileID}_{label_type}_vessels_orientation.csv", index=False)

    return final_report_df, topo_stats, orient_df


def analysis_menu():

    header = widgets.HTML(value="<b>Features selection</b>")
    box_sv = widgets.Checkbox(value=True, description='String vessel', indent=False)
    box_ns = widgets.Checkbox(value=True, description='Normal vessel', indent=False)
    box_all = widgets.Checkbox(value=True, description='Both (all)', indent=False)
    hbox_layout = widgets.Layout(display='flex', flex_flow='row')
    bbox_horizontales = widgets.HBox([box_sv, box_ns, box_all], layout=hbox_layout)
    menu_features = widgets.VBox([header, bbox_horizontales])

    def valid_seleccion(change): 
        if not box_sv.value and not box_ns.value and not box_all.value:
            box_sv.value = True

    box_sv.observe(valid_seleccion, names='value')
    box_ns.observe(valid_seleccion, names='value')
    box_all.observe(valid_seleccion, names='value')

    
    default_params = {
        "pixelDimensions": "1.0, 1.0, 1.0", 
        "pruningScale": 1, "lengthLimit": 1, "diaScale": 1, "branchingThreshold": 0.25
    }
    
    pixel_dim_widget = widgets.Text(value=default_params["pixelDimensions"], description='Pixel Dims (Z,Y,X):', style={'description_width': 'initial'})
    meta_pixel_dim_widget = widgets.Checkbox(value=True, description='Use metadata pixel dims', indent=True)
    pruning_scale_widget = widgets.FloatText(value=default_params["pruningScale"], description='Pruning Scale:', style={'description_width': 'initial'})
    length_limit_widget = widgets.FloatText(value=default_params["lengthLimit"], description='Length Limit:', style={'description_width': 'initial'})
    dia_scale_widget = widgets.FloatText(value=default_params["diaScale"], description='Dia Scale:', style={'description_width': 'initial'})
    branching_threshold_widget = widgets.FloatText(value=default_params["branchingThreshold"], description='Branching Threshold:', style={'description_width': 'initial'})

    pred_path_widget = FileChooser(Path.cwd().as_posix(), title='Select model_predictions folder', select_default=False)
    pred_path_widget.show_only_dirs = True 
    out_path_widget = FileChooser(Path.cwd().as_posix(), title='Select output root folder', select_default=False)
    out_path_widget.show_only_dirs = True

    run_button = widgets.Button(description="Run Analysis", button_style='success')
    output_area = widgets.Output()

    def on_button_click(b):
        with output_area:
            clear_output()
            print("Starting analysis...")

            params = {}
            try:
                params["pixelDimensions"] = [float(d.strip()) for d in pixel_dim_widget.value.split(',')]
            except ValueError:
                print("Error: Invalid Pixel Dimensions.")
                return

            params["pruningScale"] = pruning_scale_widget.value
            params["lengthLimit"] = length_limit_widget.value
            params["diaScale"] = dia_scale_widget.value
            params["branchingThreshold"] = branching_threshold_widget.value

            apply_sv=box_sv.value
            apply_nv=box_ns.value
            apply_all=box_all.value

            pred_path = Path(pred_path_widget.selected)
            out_path = Path(out_path_widget.selected) / 'stats'
            out_path.mkdir(parents=True, exist_ok=True)

            filenames = sorted(list(pred_path.glob("*.tiff")) + list(pred_path.glob('*.tif')))
            if not filenames:
                print("Error: No tiff files found.")
                return

            print(f"Processing {len(filenames)} files...")
            stats_list = []

            for fn in tqdm(filenames, desc="Processing"):
                try:
                    fileID = fn.stem
                    
                    try:
                        img_obj = BioImage(fn, reader=bioio_tifffile.Reader)
                        pred = img_obj.get_image_data("ZYX", C=0, T=0)
                    except:
                        img_obj = BioImage(fn)
                        pred = img_obj.get_image_data("ZYX", C=0, T=0)
                    
                    # Pixel Dims
                    current_pixel_dims = list(params["pixelDimensions"])
                    if meta_pixel_dim_widget.value:
                        pps = getattr(img_obj, "physical_pixel_sizes", None)
                        if pps:
                            z = getattr(pps, "Z", 1.0) or 1.0
                            y = getattr(pps, "Y", 1.0) or 1.0
                            x = getattr(pps, "X", 1.0) or 1.0
                            current_pixel_dims = [float(z), float(y), float(x)]
                    
                    current_params = params.copy()
                    current_params["pixelDimensions"] = current_pixel_dims

                    # Masks
                    string_vessel = (pred == 2)
                    all_vessel = (pred > 0)
                    normal_vessel = np.logical_and(all_vessel, np.logical_not(string_vessel))

                    # Volume & Metadata
                    if pred.ndim == 3:
                        num_slices, height, width = pred.shape
                    else:
                        num_slices = 1
                        height, width = pred.shape
                    
                    z_dim, y_dim, x_dim = current_pixel_dims
                    volume = num_slices * z_dim * height * y_dim * width * x_dim
                    
                    current_params.update({"num_slices": num_slices, "height": height, "width": width})

                    data_sources = {}
                    if apply_all:
                        data_sources['all'] = process_vessel_type(all_vessel, current_params, fileID, out_path, "all")
                    if apply_sv:
                        data_sources['string'] = process_vessel_type(string_vessel, current_params, fileID, out_path, "string")
                    if apply_nv:
                        data_sources['normal'] = process_vessel_type(normal_vessel, current_params, fileID, out_path, "normal")

                    stats_list.append(compile_statistics_row(fileID, volume, current_params, data_sources))
                    
                    with open(out_path / "Resume_process_files.txt", "a") as f:
                        f.write(f"{fn.name}, used_pixel_dim: {current_pixel_dims}\n")

                except Exception as e:
                    with open(out_path / "Unprocessed_files.txt", "a") as f:
                        f.write(f"{fn.name} - Error: {e}\n")

            if stats_list:
                pd.DataFrame(stats_list).to_csv(out_path.parent / "data_full_stats.csv", index=False)
                print(f"Done. Summary saved to {out_path.parent / 'data_full_stats.csv'}")

    run_button.on_click(on_button_click)
    display(menu_features,widgets.VBox([
        widgets.HTML("<b>Vessel Analysis Parameters</b>"),
        pixel_dim_widget, meta_pixel_dim_widget, pruning_scale_widget, length_limit_widget, dia_scale_widget, branching_threshold_widget,
        widgets.HTML("<b>Input/Output Paths</b>"), pred_path_widget, out_path_widget, run_button, output_area
    ]))


########################### Quality control Original #############################################################
# def plot_qc_results(df, thresholds, output_path):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
#     color_map = {
#         'Normal': '#A0A0A0',      
#         'Blob': '#E74C3C',        
#         'Noise': '#F39C12',       
#         'Empty/LowSignal': '#3498DB' 
#     }
    
#     df['Primary_Flag'] = df['QC_Flag'].apply(lambda x: x.split('|')[0] if x != '' else 'Normal')

#     for label, color in color_map.items():
#         sub = df[df['Primary_Flag'] == label]
#         ax1.scatter(sub['length_density_all'], sub['avg_thickness_all'], 
#                     c=color, label=label, alpha=0.6)
    
#     ax1.axhline(thresholds['blob'], color='red', linestyle='--', alpha=0.4)
#     ax1.set_title('Artifact Detection (Blobs)')
#     ax1.set_xlabel('Length Density')
#     ax1.set_ylabel('Avg Thickness')
#     ax1.legend()

#     for label, color in color_map.items():
#         sub = df[df['Primary_Flag'] == label]
#         ax2.scatter(sub['num_string_density'], sub['average_length_all'], 
#                     c=color, label=label, alpha=0.6)
    
#     ax2.set_title('Noise Detection (Fragmentation)')
#     ax2.set_xlabel('Num String Density')
#     ax2.set_ylabel('Average Segment Length')
#     ax2.legend()

#     plt.tight_layout()
#     plt.savefig(output_path / "QC_Outlier_Distribution.png", dpi=300)

# def run_quality_control(csv_path, output_path):
#     df = pd.read_csv(csv_path, sep=None, engine='python')
#     df.columns = df.columns.str.strip()
    
#     # Calculate Thresholds
#     metrics = {
#         'void': df['length_density_all'].quantile(0.05),
#         'high_string': df['num_string_density'].quantile(0.95),
#         'low_len': df['average_length_all'].quantile(0.25),
#         'q1_thick': df['avg_thickness_all'].quantile(0.25),
#         'q3_thick': df['avg_thickness_all'].quantile(0.75)
#     }
#     metrics['iqr_thick'] = metrics['q3_thick'] - metrics['q1_thick']
#     metrics['blob'] = metrics['q3_thick'] + (1.5 * metrics['iqr_thick'])

#     # Flagging function
#     def get_row_flags(row):
#         f = []
#         if row['length_density_all'] <= metrics['void']: f.append("Empty/LowSignal")
#         if row['num_string_density'] >= metrics['high_string'] and row['average_length_all'] <= metrics['low_len']: f.append("Noise")
#         if row['avg_thickness_all'] >= metrics['blob']: f.append("Blob")
#         return "|".join(f)

#     df['QC_Flag'] = df.apply(get_row_flags, axis=1)

#     plot_qc_results(df, metrics, output_path)

#     # Export
#     suspicious_df = df[df['QC_Flag'] != ''].copy()
#     file_col = [c for c in df.columns if 'filename' in c.lower()][0]
#     save_path = output_path / "QC_Review_List.csv"
    
#     cols_to_save = [file_col, 'QC_Flag', 'num_string_density', 'length_density_all', 'avg_thickness_all', 'average_length_all']
#     suspicious_df[cols_to_save].to_csv(save_path, index=False)
    
#     print("############################## Quality Control Complete ##############################")
#     print(f"############################## Processed {len(df)} images ##############################")
#     return df

################################### Quality control proposed cahnges ############################################
def plot_qc_results(df, thresholds, output_path):
    """
    Generates a 2x2 QC report visualizing different artifact types.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Define a consistent color map for flags
    # We use a base color for 'Normal' and distinct colors for issues
    unique_flags = df['Primary_Flag'].unique()
    palette = {
        'Normal': '#A0A0A0',      # Gray
        'Blob': '#E74C3C',        # Red
        'Noise': '#F39C12',       # Orange
        'Empty/LowSignal': '#3498DB', # Blue
        'Fragmented': '#9B59B6',  # Purple (New)
        'Tortuous/Artifact': '#2ECC71' # Green (New)
    }
    
    # Helper to plot with consistent coloring
    def scatter_plot(ax, x_col, y_col, thresh_y=None, thresh_x=None, title="", xlabel="", ylabel=""):
        for flag in unique_flags:
            subset = df[df['Primary_Flag'] == flag]
            color = palette.get(flag, '#000000') # Black fallback
            # Handle composite flags by checking substring if exact match fails
            if flag not in palette:
                if 'Blob' in flag: color = palette['Blob']
                elif 'Noise' in flag: color = palette['Noise']
            
            ax.scatter(subset[x_col], subset[y_col], c=color, label=flag, alpha=0.6, edgecolors='w', s=40)
        
        if thresh_y:
            ax.axhline(thresh_y, color='k', linestyle='--', alpha=0.5, label='Threshold')
        if thresh_x:
            ax.axvline(thresh_x, color='k', linestyle='--', alpha=0.5)
            
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.3)

    #  Blob Detection (Thickness vs Density)
    scatter_plot(axes[0, 0], 'length_density_all', 'avg_thickness_all', 
                 thresh_y=thresholds['blob_upper'], 
                 title='Artifact Detection (Blobs)', 
                 xlabel='Length Density', ylabel='Avg Thickness')

    # Noise Detection (Classic Fragmentation)
    # Note: Noise is defined by High Count AND Low Length
    scatter_plot(axes[0, 1], 'num_string_density', 'average_length_all', 
                 thresh_y=thresholds['len_lower'],
                 title='Noise Detection (Short Segments)', 
                 xlabel='String Vessel Density', ylabel='Avg Segment Length')

    # Advanced Topology (Disconnection)
    # Visualizes the ratio of endpoints to total length
    scatter_plot(axes[1, 0], 'length_density_all', 'fragmentation_index', 
                 thresh_y=thresholds['frag_upper'],
                 title='Topological Disconnection', 
                 xlabel='Length Density', ylabel='Fragmentation Index (Endpoints/Length)')

    # Shape Consistency (Straightness)
    scatter_plot(axes[1, 1], 'avg_thickness_all', 'avg_straightness_all',
                 thresh_y=thresholds['straight_lower'],
                 title='Shape Quality (Tortuosity/Jaggedness)', 
                 xlabel='Avg Thickness', ylabel='Avg Straightness')

    plt.suptitle(f"Quality Control Overview (N={len(df)})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path / "QC_Detailed_Report.png", dpi=300)
    plt.close()

def run_quality_control(csv_path, output_path):
    print("Loading data for Quality Control...")
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    
    # ---  Feature Engineering for QC ---
    # Fragmentation Index: High values indicate many endpoints per unit of length (broken network)
    # Adding 1e-9 to avoid division by zero
    df['fragmentation_index'] = df['endpoint_density_all'] / (df['length_density_all'] + 1e-9)
    
    # --- Calculate Robust Thresholds (IQR Method) ---
    def calc_bounds(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return q1, q3, q1 - 1.5 * iqr, q3 + 1.5 * iqr

    # Thickness metrics (Blobs)
    _, q3_thick, _, upper_thick = calc_bounds(df['avg_thickness_all'])
    
    # Length metrics (Noise)
    q1_len, _, lower_len, _ = calc_bounds(df['average_length_all'])
    
    # Density metrics (Empty)
    q1_den, _, lower_den, _ = calc_bounds(df['length_density_all'])
    
    # String Density (Noise trigger)
    _, q3_str_den, _, upper_str_den = calc_bounds(df['num_string_density'])

    # Straightness (Jagged artifacts have low straightness)
    q1_str, _, lower_str, _ = calc_bounds(df['avg_straightness_all'])
    
    # Fragmentation (Broken networks)
    _, q3_frag, _, upper_frag = calc_bounds(df['fragmentation_index'])

    # Compile thresholds dictionary for plotting/logging
    metrics = {
        'blob_upper': upper_thick,
        'len_lower': max(lower_len, 0), # Length can't be negative
        'void_lower': max(df['length_density_all'].quantile(0.05), 0), # Keep 5th percentile for hard cut
        'high_string_den': df['num_string_density'].quantile(0.95), # Keep 95th percentile
        'straight_lower': lower_str,
        'frag_upper': upper_frag
    }
    
    # ---  Flagging Logic ---
    def get_row_flags_and_score(row):
        flags = []
        score = 0
        
        # A. Empty / Low Signal
        # Logic: Very little vessel length found in volume
        if row['length_density_all'] <= metrics['void_lower']:
            flags.append("Empty/LowSignal")
            score += 3 # High severity
            
        # B. Noise (Dust/Debris)
        # Logic: High density of objects BUT very short average length
        if (row['num_string_density'] >= metrics['high_string_den']) and \
           (row['average_length_all'] <= metrics['len_lower']):
            flags.append("Noise")
            score += 2
            
        # C. Blobs (Artifacts)
        # Logic: Average thickness is statistically an outlier
        if row['avg_thickness_all'] >= metrics['blob_upper']:
            flags.append("Blob")
            score += 2
            
        # D. Fragmented (Topological Issues) - NEW
        # Logic: Too many endpoints relative to length
        if row['fragmentation_index'] >= metrics['frag_upper']:
            flags.append("Fragmented")
            score += 1
            
        # E. Tortuous/Jagged - NEW
        # Logic: Vessels are unnaturally wiggly (segmentation edges) or failing geometric check
        if row['avg_straightness_all'] <= metrics['straight_lower']:
            flags.append("Tortuous/Artifact")
            score += 1

        return "|".join(flags), score

    # Apply logic
    df[['QC_Flag', 'QC_Severity']] = df.apply(
        lambda row: pd.Series(get_row_flags_and_score(row)), axis=1
    )
    
    # Create a Primary Flag for simpler visualization (picks the first flag found)
    df['Primary_Flag'] = df['QC_Flag'].apply(lambda x: x.split('|')[0] if x != '' else 'Normal')

    # --- 4. Generate Outputs ---
    plot_qc_results(df, metrics, output_path)

    # Export suspicious files sorted by Severity (Worst files on top)
    cols_to_save = [
        'filename', 'QC_Flag', 'QC_Severity', 
        'num_string_density', 'length_density_all', 
        'avg_thickness_all', 'fragmentation_index', 'avg_straightness_all'
    ]
    
    # Filter only flagged files
    suspicious_df = df[df['QC_Severity'] > 0].sort_values(by='QC_Severity', ascending=False)
    
    review_path = output_path / "QC_Review_List.csv"
    suspicious_df[cols_to_save].to_csv(review_path, index=False)
    
    # Also save the full DF with the new metrics for further analysis
    df.to_csv(output_path / "data_full_stats_with_qc.csv", index=False)
    
    print("############################## Quality Control Complete ##############################")
    print(f"Review list saved to: {review_path}")
    
    return df

def qc_menu():
    
    csv_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select the .csv stats file',
        select_default=False 
    )
    csv_path_widget.filter_pattern = ['*.csv'] 
    csv_path_widget.layout = widgets.Layout(width='80%')

    run_button = widgets.Button(description="Run Quality Control", button_style='success')
    
    output_area = widgets.Output()
    def on_button_click(b):
        with output_area:
            clear_output()
            print("############################## Starting Quality Control ##############################")
            if csv_path_widget.selected:
                input_path = Path(csv_path_widget.selected)
                out_path = input_path.parent / 'quality_control_output'
                out_path.mkdir(parents=True, exist_ok=True)
                run_quality_control(input_path, out_path)
                print(f" Results saved on -> {out_path}")
            else:
                print("Please Select a csv file.")
                return


    run_button.on_click(on_button_click)
    display(widgets.VBox([widgets.HTML("<b>Quality Control Parameters</b>"),csv_path_widget, run_button, output_area]))