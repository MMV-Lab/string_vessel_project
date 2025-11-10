import os
import numpy as np
from pathlib import Path
import pandas as pd
from bioio import BioImage
import ipywidgets as widgets
from IPython.display import display, clear_output
from tqdm.notebook import tqdm
import bioio_tifffile
from ipyfilechooser import FileChooser
import trimesh
from bioio.writers import OmeTiffWriter
import vedo
import napari




def mesh_compute(volume,pixel_size,out_path,fileID,smooth_use):
    sv_vol = volume.copy()
    sv_vol[sv_vol!=2] = 0
    
    nv_vol = volume.copy()
    nv_vol[nv_vol!=1] = 0
    
    if np.max(sv_vol) != 0:
        sv_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(sv_vol, pitch=pixel_size)
        sv_vedo = vedo.Mesh([sv_mesh.vertices, sv_mesh.faces])
        sv_path = out_path / f'{fileID}_sv_mesh.stl'
        vedo.write(sv_vedo, sv_path)
        if smooth_use:
            sv_smooth = sv_vedo.smooth(niter=20, pass_band=0.07)
            sv_path_smooth = out_path / f'{fileID}_sv_mesh_smooth.stl'
            vedo.write(sv_smooth, sv_path_smooth)
      
    if np.max(nv_vol) != 0:
        nv_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(nv_vol, pitch=pixel_size)
        nv_vedo = vedo.Mesh([nv_mesh.vertices, nv_mesh.faces])
        nv_path = out_path / f'{fileID}_nv_mesh.stl'
        vedo.write(nv_vedo, nv_path)

        if smooth_use:
            nv_smooth = nv_vedo.smooth(niter=20, pass_band=0.07)
            nv_path_smooth = out_path / f'{fileID}_nv_mesh_smooth.stl' 
            vedo.write(nv_smooth, nv_path_smooth)

    



def mesh_menu():
    """
    Provides an interactive menu with editable text fields to set parameters
    and run mesh generation.
    """

    # Default parameters
    default_params = {
        "pixelDimensions": "1.0, 1.0, 1.0"
    }
 
    # Create widgets for each parameter
    pixel_dim_widget = widgets.Text(
        value=default_params["pixelDimensions"],
        description='Pixel Dimensions:',
        layout=widgets.Layout(description_width='initial'),
        style={'description_width': '40%'}
    )

    # Path widgets

    pred_path_widget = FileChooser(
        Path.cwd().as_posix(), 
        title='Select the model_predictions folder path',
        select_default=False 
    )
    pred_path_widget.show_only_dirs = True 
    pred_path_widget.layout = widgets.Layout(width='70%')

    smooth_use_checkbox = widgets.Checkbox(
        value=False,
        description=" Smooth Mesh",
        disabled=False
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
                params["pixelDimensions"] = tuple([float(d.strip()) for d in pixel_dim_widget.value.split(',')])
            except ValueError:
                print("Invalid input for Pixel Dimensions. Please use a comma-separated list of numbers.")
                return

            for key, value in params.items():
                print(f"  {key}: {value}")

            pred_path_str = pred_path_widget.selected
    
            if not pred_path_str :
                raise ValueError("Please select input folder")
                return

            pred_path = Path(pred_path_widget.value)
            mesh_out = pred_path.parent / 'mesh_vect_out'
            mesh_out.mkdir(parents=True, exist_ok=True)

            smooth_use = smooth_use_checkbox.value

            print(f" Segmentation Results Path: {pred_path}")
            print(f" Save Mesh results Path: {mesh_out}")
        

            filenames = sorted(pred_path.glob("*.tiff"))
            filenames.extend(list(pred_path.glob('*.tif')))
            if not filenames:
                raise ValueError(f"No .tiff files found in {pred_path}. Please check the path and file extensions.")
                return

        
            print(f"{len(filenames)} files found to be analyzed")
            for fn in tqdm(filenames, desc= "Analysis file progress "):
                fileID = fn.stem
                try:
                    pred = BioImage(fn, reader=bioio_tifffile.Reader).get_image_data("ZYX", C=0, T=0)
                except Exception as e:
                    try:
                        pred = BioImage(fn).get_image_data("ZYX", C=0, T=0)
                    except Exception as e:
                        raise ValueError("Error at reading time.")
                opp = mesh_out / fileID
                opp.mkdir(parents=True, exist_ok=True)
                mesh_compute(pred,params["pixelDimensions"],opp,fileID,smooth_use)
    run_button.on_click(on_button_click)

    # Display the widgets
    display(
        widgets.VBox([
            widgets.HTML("<b>Mesh Analysis Parameters</b>"),
            pixel_dim_widget,
            widgets.HTML("<b>Input Path</b>"),
            pred_path_widget,
            smooth_use_checkbox,
            run_button,
            output_area
        ])
    )


def analyze_single_component(comp_mesh, component_id, mesh_type, fileID, ref_z=np.array([0.0, 0.0, 1.0]), ref_x=np.array([1.0, 0.0, 0.0])):
    """Calculates vector statistics and volume for a single connected component."""
    volume = comp_mesh.volume()
    comp_normals = comp_mesh.compute_normals(points=False).vertex_normals
    comp_tangents, _ = calculate_tbn_safe(comp_normals, ref_z, ref_x)
    normal_mean = np.mean(comp_normals, axis=0)
    normal_std = np.std(comp_normals, axis=0)
    tan_mean = np.mean(comp_tangents, axis=0)
    tan_std = np.std(comp_tangents, axis=0)

    results = {
        'image_id': fileID, 
        'vessel': f'comp_{component_id}', 
        'volume': volume, 
        'normal_mean': f'({normal_mean[0]:.6f}, {normal_mean[1]:.6f}, {normal_mean[2]:.6f})',
        'normal_std': f'({normal_std[0]:.6f}, {normal_std[1]:.6f}, {normal_std[2]:.6f})',
        'tan_mean': f'({tan_mean[0]:.6f}, {tan_mean[1]:.6f}, {tan_mean[2]:.6f})',
        'tan_std': f'({tan_std[0]:.6f}, {tan_std[1]:.6f}, {tan_std[2]:.6f})',
    }

    return results


def component_analysis(mesh, mesh_type, base_path, fileID):
    """
    Splits the mesh into connected components and calculates statistics for each.
    Returns the list of statistics dictionaries.
    """

    components = mesh.split()
    all_stats = []

    ref_z = np.array([0.0, 0.0, 1.0])
    ref_x = np.array([1.0, 0.0, 0.0])

    for i, comp_mesh in enumerate(components):
        stats = analyze_single_component(comp_mesh, i + 1, mesh_type, fileID, ref_z, ref_x)
        all_stats.append(stats)

    return all_stats

def save_global_stats(all_sv_stats, all_nv_stats, base_path):
    """Saves the accumulated component statistics to global CSV files."""

    col_order = ['image_id', 'vessel', 'volume', 'normal_mean', 'normal_std', 'tan_mean', 'tan_std']
    
    # Guardar NV stats
    if all_nv_stats:
        df_nv_stats = pd.DataFrame(all_nv_stats)
        csv_nv_out = base_path / 'Summary_nv_vectors_stats.csv'
        df_nv_stats = df_nv_stats[col_order]
        df_nv_stats.to_csv(csv_nv_out, index=False)
        

    # Guardar SV stats
    if all_sv_stats:
        df_sv_stats = pd.DataFrame(all_sv_stats)
        csv_sv_out = base_path / 'Summary_sv_vectors_stats.csv'
        df_sv_stats = df_sv_stats[col_order]
        df_sv_stats.to_csv(csv_sv_out, index=False)
        
        

def calculate_tbn_safe(normals, ref_primary, ref_secondary, epsilon=1e-6):
    
    reference_primary = np.tile(ref_primary, (len(normals), 1))
    bitangents = np.cross(normals, reference_primary)
    bitangent_sq_norm = np.sum(bitangents**2, axis=1)
    is_parallel = bitangent_sq_norm < epsilon**2 
    
    if np.any(is_parallel):
        reference_secondary = np.tile(ref_secondary, (len(normals), 1))
        alternative_bitangents = np.cross(normals[is_parallel], reference_secondary[is_parallel])
        bitangents[is_parallel] = alternative_bitangents

    bitangents = bitangents / np.linalg.norm(bitangents, axis=1, keepdims=True)
    tangents = np.cross(bitangents, normals) 
    

    return tangents, bitangents

def compute_vector(sv,nv, base_path,fileID):
    
    ref_z = np.array([0.0, 0.0, 1.0]) 
    ref_x = np.array([1.0, 0.0, 0.0])
    
    if sv is not None:
        sv_normal = sv.compute_normals(points=True, cells=True)
        sv_points = sv_normal.vertices
        sv_directions = sv_normal.vertex_normals
        sv_norm = np.stack([sv_points, sv_directions], axis=1)
        sv_out =  base_path / f'{fileID}_sv_normal.npy'
        np.save(sv_out, sv_norm )
        sv_tangents, sv_bitangents = calculate_tbn_safe(sv_directions, ref_z, ref_x)
        sv_tangent = np.stack([sv_points, sv_tangents], axis=1)
        svt_out =  base_path / f'{fileID}_sv_tang.npy'
        np.save(svt_out, sv_tangent )
    
    if nv is not None:
        nv_normal = nv.compute_normals(points=True, cells=True)
        nv_points = nv_normal.vertices
        nv_directions = nv_normal.vertex_normals
        nv_norm = np.stack([nv_points, nv_directions], axis=1)
        nv_out = base_path /  f'{fileID}_nv_normal.npy'
        np.save(nv_out, nv_norm )
        nv_tangents, nv_bitangents = calculate_tbn_safe(nv_directions, ref_z, ref_x)
        nv_tangent = np.stack([nv_points, nv_tangents], axis=1)
        nvt_out = base_path /  f'{fileID}_nv_tang.npy'
        np.save(nvt_out, nv_tangent )


def vector_menu():
    """
    Provides an interactive menu with editable text fields to set parameters
    and run the vector generation.
    """

  
    pred_path_widget = FileChooser(
        Path.cwd().as_posix(), 
        title='Select the mesh_vect_out folder path',
        select_default=False 
    )
    pred_path_widget.show_only_dirs = True 
    pred_path_widget.layout = widgets.Layout(width='70%')

    mesh_use_dropdown = widgets.Dropdown(
        options={
            'Normal': 'normal', 
            'Smooth': 'smooth'
        },
        value='normal', 
        description='Used Mesh:',
        disabled=False,
        style={'description_width':'50%'} 
    )

    run_button = widgets.Button(
        description="Run Analysis",
        button_style='success', 
    )
    output_area = widgets.Output()

    def on_button_click(b):
        with output_area:
            clear_output()
            pred_path_str = pred_path_widget.selected
            mesh_type = mesh_use_dropdown.value

            if not pred_path_str :
                raise ValueError("Please select input folder")
                return
            
            pred_path = Path(pred_path_widget.value)

            all_nv_stats = []
            all_sv_stats = []
            
            for element in tqdm(pred_path.iterdir(),desc= "Vector generation in file "):
                if element.is_dir():
                    file = str(element.name)
                    nv_mesh = None
                    sv_mesh = None
                    
                    if mesh_type == 'smooth':
                        if (element / f'{file}_nv_mesh_smooth.stl').is_file():
                            nv_mesh = vedo.load(element /f'{file}_nv_mesh_smooth.stl')
                        else:
                            if (element / f'{file}_nv_mesh.stl').is_file():
                                nv_mesh = vedo.load(element /f'{file}_nv_mesh.stl')

                        if (element / f'{file}_sv_mesh_smooth.stl').is_file():
                            sv_mesh = vedo.load(element /f'{file}_sv_mesh_smooth.stl')
                        else:
                            if(element / f'{file}_sv_mesh.stl').is_file():
                                sv_mesh = vedo.load(element /f'{file}_sv_mesh.stl')
                    else:
                        
                        if (element / f'{file}_nv_mesh.stl').is_file():
                            nv_mesh = vedo.load(element /f'{file}_nv_mesh.stl')
                        
                        if (element / f'{file}_sv_mesh.stl').is_file():
                            sv_mesh = vedo.load(element /f'{file}_sv_mesh.stl')
                    
                    
                    if sv_mesh is None and nv_mesh is None:
                        with open(pred_path / Path("Computation_log.txt"), "a") as f:
                            f.write(f"{file} no .stl files found.\n")
                    else:     
                        if sv_mesh is not None:
                            stats = component_analysis(sv_mesh, 'sv', element, file)
                            all_sv_stats.extend(stats)
                        else:
                            with open(pred_path / Path("Computation_log.txt"), "a") as f:
                                f.write(f"{file} string vessels (sv) .stl file missing.\n")   

                        if nv_mesh is not None:
                            stats = component_analysis(nv_mesh, 'nv', element, file)
                            all_nv_stats.extend(stats)
                        else:
                            with open(pred_path / Path("Computation_log.txt"), "a") as f:
                                f.write(f"{file} normal vessels (nv) .stl file missing.\n") 
                        
                    compute_vector(sv_mesh,nv_mesh,element,file)
        
            save_global_stats(all_sv_stats, all_nv_stats, pred_path)
            print("Computation Finished")
            print(f"Vector csv files saved at {pred_path}")
            if os.path.exists(pred_path / Path("Computation_log.txt")):
                print('#####################################WARNING#############################')
                print('For some images .stl files are missing and the vector computations are omitted')
                print('The deatails are on the log file:')
                print(pred_path / Path("Computation_log.txt"))

            

    run_button.on_click(on_button_click)

    # Display the widgets
    display(
        widgets.VBox([
            widgets.HTML("<b>Vector Computation Parameters</b>"),
            widgets.HTML("<b>Input Path</b>"),
            pred_path_widget,
            mesh_use_dropdown,
            run_button,
            output_area
        ])
    )

def visualizer(base_path):
    file_id = str(base_path.name)
    sv_file = base_path / f'{file_id}_sv_normal.npy'
    nv_file = base_path / f'{file_id}_nv_normal.npy'

    svt_file = base_path / f'{file_id}_sv_tang.npy'
    nvt_file = base_path / f'{file_id}_nv_tang.npy'

    viewer = napari.Viewer()


    try:
        sv_norm_data = np.load(sv_file)

        viewer.add_vectors(
            sv_norm_data,
            name=f'{file_id}_sv_norm',
            length=5,         
            edge_width=0.5,     
            edge_color='lime'
        )
    except FileNotFoundError:
        print(f"Error: File not found {sv_file}")


    try:
        nv_norm_data = np.load(nv_file)
        

        viewer.add_vectors(
            nv_norm_data,
            name=f'{file_id}_nv_norm',
            length=5,
            edge_width=0.5,
            edge_color='#ff557fff'
        )
    except FileNotFoundError:
        print(f"Error: File not found {nv_file}")


    try:
        svt_norm_data = np.load(svt_file)
        viewer.add_vectors(
            svt_norm_data,
            name=f'{file_id}_sv_tan',
            length=5,         
            edge_width=0.5,     
            edge_color='yellow'
        )
    except FileNotFoundError:
        print(f"Error: File not found {svt_file}")

    try:
        nvt_norm_data = np.load(nvt_file)

        viewer.add_vectors(
            nvt_norm_data,
            name=f'{file_id}_nv_tan',
            length=5,         
            edge_width=0.5,     
            edge_color='#aa00ffff'
        )
    except FileNotFoundError:
        print(f"Error: File not found {nvt_file}")

    napari.run()
    

def visualizer_menu():
    """
    Provides an interactive menu with editable text fields to set parameters
    and run the visualization for the vectors.
    """
    pred_path_widget = FileChooser(
        Path.cwd().as_posix(), 
        title='Select the path tho the desired .npy files',
        select_default=False 
    )
    pred_path_widget.show_only_dirs = True 
    pred_path_widget.layout = widgets.Layout(width='70%')

    run_button = widgets.Button(
        description="Run Visualization",
        button_style='success', 
    )
    output_area = widgets.Output()

    def on_button_click(b):
        with output_area:
            clear_output()
            pred_path_str = pred_path_widget.selected

            if not pred_path_str :
                raise ValueError("Please select input folder")
                return
            
            pred_path = Path(pred_path_widget.value)
            file = str(pred_path.name)
            deploy = False
            if (pred_path / f'{file}_nv_normal.npy').is_file():
                deploy = True
            elif (pred_path / f'{file}_nv_tang.npy').is_file():
                deploy = True    
            elif (pred_path / f'{file}_sv_normal.npy').is_file():
                deploy = True
            elif(pred_path / f'{file}_sv_normal.npy').is_file():
                deploy = True    
            else:
                raise ValueError("Error reading .npy file, No files found.")
            
            if deploy:
                visualizer(pred_path)
                
                  
            

    run_button.on_click(on_button_click)

    # Display the widgets
    display(
        widgets.VBox([
            widgets.HTML("<b>Vector Visualization</b>"),
            widgets.HTML("<b>Input Path</b>"),
            pred_path_widget,
            run_button,
            output_area
        ])
    )