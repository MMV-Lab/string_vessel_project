#This code is an adaptation for the code: https://github.com/TopoXLab/VesselAnalysis?tab=readme-ov-file
#We implement grapg generation and numba aceleration
#Related to the paper: https://arxiv.org/abs/2402.16894

import os
from PIL import Image
import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt as distrans
import tcripser
from gudhi.representations import vector_methods
import matplotlib.pyplot as plt
import persim
from numba import jit, cuda
from tqdm import tqdm
import math
from IPython.display import display, HTML
import ipywidgets as widgets
from ipywidgets import Layout

def extract_vessel_features(featurenames=['betti_PHT', 'betti', 'PI', 'PI_local'],
                            root='/root/path/to/output/',
                            vesselpath='/path/to/segmentations/',
                            save_npy=True,
                            generate_images=True,
                            plot_per_patch=False,
                            tile_size=(8,8,8)): 


    vessel = sorted([f for f in os.listdir(vesselpath)])
    saveroot_npy_base = os.path.join(root,'TDA_out','npy_files')
    saveroot_img_base = os.path.join(root,'TDA_out','graphs')

    print(f"{len(vessel)} Files found to process...")
    print("Starting Topological feature extraction")

    if save_npy and not os.path.isdir(saveroot_npy_base):
        os.makedirs(saveroot_npy_base)

    if generate_images and not os.path.isdir(saveroot_img_base):
        os.makedirs(saveroot_img_base)

    for v in vessel:
        name = v.replace('.tiff', '')
        current_npy_folder = os.path.join(saveroot_npy_base, name)
        current_img_folder = os.path.join(saveroot_img_base, name)

        if save_npy and not os.path.isdir(current_npy_folder):
            os.makedirs(current_npy_folder)

        if generate_images and not os.path.isdir(current_img_folder):
            os.makedirs(current_img_folder)

        if generate_images and plot_per_patch:
            current_patch_img_folder = os.path.join(current_img_folder, 'patch_graphs')
            if not os.path.isdir(current_patch_img_folder):
                os.makedirs(current_patch_img_folder)

    if 'betti_PHT' in featurenames:
        Betti_PHT(vessel, vesselpath, saveroot_npy_base, saveroot_img_base, save_npy, generate_images)
    if 'betti' in featurenames:
        Betti_ending(vessel, vesselpath, saveroot_npy_base, saveroot_img_base, save_npy, generate_images)
    if  'PI' in featurenames:
        PI_ending(vessel, vesselpath, saveroot_npy_base, saveroot_img_base, save_npy, generate_images)
    if  'PI_local' in featurenames:
        PI_Local(vessel, vesselpath, saveroot_npy_base, saveroot_img_base, save_npy, generate_images, plot_per_patch, tile_size) 


def PI_Local(masks, vesselpath, saveroot_npy, saveroot_img, save_npy, generate_images, plot_per_patch, tile_size): 
    print("Computing Patchwise Persistence Image")
    gen_PI = vector_methods.PersistenceImage(bandwidth=1.0, weight=lambda x: x[1]/0.6 if x[1]<0.6 else 1, resolution=[20, 20], im_range=[0,1,0,1])
    
   
    tile_divisor_k, tile_divisor_h, tile_divisor_w = tile_size 

    for f in tqdm(masks, desc= "Processing file"):
        pds = []
        name = f.split('.')[0]

        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(vesselpath, f))).astype(np.float32)
        # Break 3D image into tiles
        k_img_dim, h_img_dim, w_img_dim = image.shape
        
        # Calculate K, M, N (actual tile dimensions) 
        K = k_img_dim // tile_divisor_k
        M = h_img_dim // tile_divisor_h
        N = w_img_dim // tile_divisor_w
        
        # Adjust image dimensions to be multiples of the tile divisors
        k_img_dim = k_img_dim - (k_img_dim % tile_divisor_k)
        h_img_dim = h_img_dim - (h_img_dim % tile_divisor_h)
        w_img_dim = w_img_dim - (w_img_dim % tile_divisor_w)
        
        tiles = [image[z:z+K, x:x+M, y:y+N] for z in range(0,k_img_dim,K) for x in range(0,h_img_dim,M) for y in range(0,w_img_dim,N)]

        patch_graphs_folder = os.path.join(saveroot_img, name, 'patch_graphs')
        if generate_images and plot_per_patch and not os.path.isdir(patch_graphs_folder):
            os.makedirs(patch_graphs_folder)

        for i, tile in enumerate(tiles):
            if np.sum(tile):
                skeleton = skeletonize(tile, method='lee')
                origins = find_one_degrees(skeleton)
                if len(origins[0]):
                    distance_map = bfs_distance(tile, origins, metric='geodesic')
                else:
                    distance_map = tile/255
            else:
                distance_map = tile/255
            distance_map = np.where(distance_map<0, 1, distance_map)
            pd = get_PD(distance_map, name)


            if pd.size > 0:
                pds.append(pd)

                # Generate persistence diagram for the patch 
                if generate_images and plot_per_patch:
                    fig, ax = plt.subplots(1, 1)
                    persim.plot_diagrams([pd], ax=ax, show=False)
                    ax.set_title('Persistence Diagram')
                    plt.savefig(os.path.join(patch_graphs_folder, f'pi_local_persistence_diagram_tile_{i}.png'))
                    plt.close(fig)


        if len(pds) > 0:
            pis = gen_PI.transform(pds)

            if save_npy:
                np.save(os.path.join(saveroot_npy, name, 'pi_local_persistence_diagram.npy'), pis)

            # Averages all PIs from the patches generating a headmap
            if generate_images:
                avg_pi = np.mean(pis, axis=0)
                fig, ax = plt.subplots()
                ax.imshow(avg_pi.reshape(20, 20), cmap='viridis', origin='lower')
                ax.set_title('Avg. PI (Local)')
                plt.colorbar(ax.imshow(avg_pi.reshape(20, 20), cmap='viridis', origin='lower'))
                plt.savefig(os.path.join(saveroot_img, name, 'pi_local_avg.png'))
                plt.close(fig)

def Betti_ending(masks, vesselpath, saveroot_npy, saveroot_img, save_npy, generate_images):
    print('Computing final Betti curve')
    for f in tqdm(masks, desc= "Processing file"):
        name = f.split('.')[0]
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(vesselpath, f))).astype(np.float32)

        skeleton = skeletonize(image, method='lee')
        origins = find_one_degrees(skeleton)
        if len(origins[0]):
            distance_map = bfs_distance(image, origins, metric='geodesic')
        else:
            distance_map = image
        distance_map = np.where(distance_map<0, 1, distance_map)
        betti = get_betticurve(distance_map, name)

        if save_npy:
            np.save(os.path.join(saveroot_npy, name, 'betti_final.npy'), betti)

        # Generate final Betti curve plot
        if generate_images:
            filtration = np.arange(0, 1, 0.01)
            betti_curve0 = betti[:len(filtration)]
            betti_curve1 = betti[len(filtration):]

            fig, ax = plt.subplots()
            ax.plot(filtration, betti_curve0, label='Betti 0 (Connected Components)')
            ax.plot(filtration, betti_curve1, label='Betti 1 (Loops)')
            ax.set_xlabel('Filtration Value')
            ax.set_ylabel('Number of Features')
            ax.set_title('Final Betti Curve')
            ax.legend()
            plt.grid(True)
            plt.savefig(os.path.join(saveroot_img, name, 'betti_final.png'))
            plt.close(fig)

def Betti_PHT(masks, vesselpath, saveroot_npy, saveroot_img, save_npy, generate_images):
    print('Computing Betti PHT curve for filtrations')
    directions = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
    direction_names = ['x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz']

    for f in tqdm(masks, desc="Processing file"):
        bettis = []
        name = f.split('.')[0]

        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(vesselpath, f))).astype(np.float32)

        if generate_images:
            fig_pht, ax_pht = plt.subplots(len(directions), 1, figsize=(8, 4 * len(directions)))
            fig_pht.suptitle(f'Betti Curves for Filtrations of {name}', y=1.02)
            filtration = np.arange(0, 1, 0.01)

        
        for i, v in enumerate(directions):
            
            if cuda.is_available():
                distance_map = scan_cuda(image, np.array(v, dtype=np.float32))
            else:
                distance_map = scan(image, np.array(v, dtype=np.float32))
            
            betti = get_betticurve(distance_map, name)
            bettis.append(betti)

            if generate_images:
                betti_curve0 = betti[:len(filtration)]
                betti_curve1 = betti[len(filtration):]

                
                current_ax = ax_pht[i] if len(directions) > 1 else ax_pht
                current_ax.plot(filtration, betti_curve0, label='Betti 0')
                current_ax.plot(filtration, betti_curve1, label='Betti 1')
                current_ax.set_title(f'Direction: {direction_names[i]} ({v})')
                current_ax.set_xlabel('Filtration Value')
                current_ax.set_ylabel('Number of Features')
                current_ax.legend()
                current_ax.grid(True)

        bettis = np.asarray(bettis).flatten()

        if save_npy:
            np.save(os.path.join(saveroot_npy, name, 'betti_filtartions.npy'), bettis)

        if generate_images:
            plt.tight_layout()
            plt.savefig(os.path.join(saveroot_img, name, 'betti_filtrations.png'))
            plt.close(fig_pht)

@jit(nopython=True)
def scan(image, v):
    '''
    Generate distance map for a given direction (CPU version)
    '''
    k,h,w = image.shape
    distance_map = np.zeros((k,h,w), dtype=np.float32)
    for i in range(k):
        for j in range(h):
            for l in range(w):
                dis = v[0]*(i+1) + v[1]*(j+1) + v[2]*(l+1)
                distance_map[i,j,l] = dis
    distance_map = np.where(image==0, -1.0, distance_map)
    max_dis = np.max(distance_map)
    if max_dis == 0:
        return distance_map
    distance_map = np.where(distance_map<0, 1.0, distance_map/max_dis)
    return distance_map

@cuda.jit
def cuda_scan_kernel(image, v_gpu, distance_map_out):
    # Calculate the global index of the thread
    k_idx, h_idx, w_idx = cuda.grid(3)

    # Ensure the index is within image bounds
    if k_idx < image.shape[0] and h_idx < image.shape[1] and w_idx < image.shape[2]:
        if image[k_idx, h_idx, w_idx] != 0:
            dis = v_gpu[0] * (k_idx + 1) + v_gpu[1] * (h_idx + 1) + v_gpu[2] * (w_idx + 1)
            distance_map_out[k_idx, h_idx, w_idx] = dis
        else:
            distance_map_out[k_idx, h_idx, w_idx] = -1.0

def scan_cuda(image, v):
    '''
    Generate distance map for a given direction (GPU version)
    '''
    k, h, w = image.shape
    distance_map = np.zeros((k, h, w), dtype=np.float32)

    # Define the dimensions of the grid and the block
    threadsperblock = (8, 8, 8)
    blockspergrid_x = math.ceil(k / threadsperblock[0])
    blockspergrid_y = math.ceil(h / threadsperblock[1])
    blockspergrid_z = math.ceil(w / threadsperblock[2])
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)


    d_image = cuda.to_device(image)
    d_v = cuda.to_device(v)
    d_distance_map = cuda.device_array_like(distance_map)

    # Launch the kernel
    cuda_scan_kernel[blockspergrid, threadsperblock](d_image, d_v, d_distance_map)
    cuda.synchronize()


    distance_map = d_distance_map.copy_to_host()

    max_dis = np.max(distance_map)
    if max_dis == 0:
        return distance_map

    distance_map = np.where(distance_map < 0, 1.0, distance_map / max_dis)
    return distance_map

def PI_ending(masks, vesselpath, saveroot_npy, saveroot_img, save_npy, generate_images):
    print("Computing Persistence Image")
    gen_PI = vector_methods.PersistenceImage(bandwidth=1.0, weight=lambda x: x[1]/0.6 if x[1]<0.6 else 1, resolution=[20, 20], im_range=[0,1,0,1])
    for f in tqdm(masks, desc= "Processing file"):
        pds = []
        name = f.split('.')[0]

        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(vesselpath, f))).astype(np.float32)

        skeleton = skeletonize(image, method='lee')
        origins = find_one_degrees(skeleton)
        if len(origins[0]):
            distance_map = bfs_distance(image, origins, metric='geodesic')
        else:
            distance_map = image
        distance_map = np.where(distance_map<0, 1, distance_map)
        pd = get_PD(distance_map, name)
        pds.append(pd)
        pis = gen_PI.transform(pds)

        if save_npy:
            np.save(os.path.join(saveroot_npy, name, 'pd_final_persistence_image.npy'), pis)

        if generate_images:
            fig, ax = plt.subplots(1, 1)
            persim.plot_diagrams([pd], ax=ax, show=False)
            ax.set_title('Final Persistence Diagram')
            plt.savefig(os.path.join(saveroot_img, name, 'pd_final_persistence_diagram.png'))
            plt.close(fig)

            fig_pi, ax_pi = plt.subplots()
            ax_pi.imshow(pis[0].reshape(20, 20), cmap='viridis', origin='lower')
            ax_pi.set_title('Final Persistence Image')
            plt.colorbar(ax_pi.imshow(pis[0].reshape(20, 20), cmap='viridis', origin='lower'))
            plt.savefig(os.path.join(saveroot_img, name, 'pd_final_persistence_image.png'))
            plt.close(fig_pi)

@jit(nopython=True)
def find_one_degrees(image):
    '''
    Function to find one-degree points in a skeleton (CPU version)
    '''
    z,h,w = image.shape
    positions = [np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)]

    for k_idx in range(z):
        for i_idx in range(h):
            for j_idx in range(w):
                if image[k_idx, i_idx, j_idx] == 1:
                    if _degree(image, k_idx, i_idx, j_idx) == 1:
                        # Append operations can be slow for large arrays, consider pre-allocating or using lists and converting to array at end
                        positions[0] = np.append(positions[0], k_idx)
                        positions[1] = np.append(positions[1], i_idx)
                        positions[2] = np.append(positions[2], j_idx)
    return positions

@jit(nopython=True)
def _degree(image, k, i, j):
    '''
    Function to calculate the degree of a point in a skeleton (CPU version)
    '''
    D,H,W = image.shape
    sum_neighbors = 0
    for kk in range(max(0, k-1), min(D, k+2)):
        for ii in range(max(0, i-1), min(H, i+2)):
            for jj in range(max(0, j-1), min(W, j+2)):
                if (kk, ii, jj) != (k, i, j) and image[kk, ii, jj] == 1:
                    sum_neighbors += 1
    return sum_neighbors

@jit(nopython=True)
def bfs_distance(skeleton, origins, metric='geodesic'):
    '''
    Function to calculate distance map from a set of origins (CPU version)
    '''
    D,H,W = skeleton.shape

    map = -np.ones((D, H, W), dtype=np.float32)

    visited = np.zeros((D, H, W), dtype=np.bool_)

    q = []

    for i in range(len(origins[0])):
        x, y, z = origins[0][i], origins[1][i], origins[2][i]
        if skeleton[x][y][z] == 1:
            map[x][y][z] = 0
            visited[x,y,z] = True
            q.append((x,y,z))

    head = 0
    while head < len(q):
        i, j, k = q[head]
        head += 1

        for ni in range(max(0, i-1), min(D, i+2)):
            for nj in range(max(0, j-1), min(H, j+2)):
                for nk in range(max(0, k-1), min(W, k+2)):
                    if (ni, nj, nk) != (i, j, k) and (not visited[ni, nj, nk]) and skeleton[ni, nj, nk] == 1:
                        visited[ni, nj, nk] = True
                        q.append((ni, nj, nk))
                        if metric == 'geodesic':
                            map[ni][nj][nk] = map[i][j][k] + 1

    max_dis = np.max(map)
    if max_dis == 0 and np.any(map > -1):
        return map
    elif max_dis == 0 and not np.any(map > -1):
        return (skeleton/255).astype(np.float32)

    map = np.where(map < 0, np.float32(1), map / np.float32(max_dis))
    return map

def get_PD(map, name):
    '''
    Function to calculate persistence diagram from a distance map
    '''
    pd = tcripser.computePH(map, maxdim=1)
    pd = pd[pd[:,2] <= 1, 1:3]
    return pd

def get_betticurve(map, name):
    '''
    Function to calculate betti curve from a distance map
    '''
    pd = tcripser.computePH(map, maxdim=1)
    filtration = np.arange(0, 1, 0.01)
    betti_curve0 = []
    betti_curve1 = []

    for f_val in filtration:
        betti0_count = np.sum((pd[:,0] == 0) & (pd[:,1] <= f_val) & (pd[:,2] >= f_val))
        betti_curve0.append(betti0_count)
        betti1_count = np.sum((pd[:,0] == 1) & (pd[:,1] <= f_val) & (pd[:,2] >= f_val))
        betti_curve1.append(betti1_count)

    betti_curve0, betti_curve1 = np.asarray(betti_curve0), np.asarray(betti_curve1)
    betti = np.hstack((betti_curve0, betti_curve1))
    return betti


def generate_topological_menu():

    pi_local_checkbox = widgets.Checkbox(value=True, description='PI_local', indent=False)
    pi_checkbox = widgets.Checkbox(value=True, description='PI', indent=False)

    feature_names_checkboxes = [
        widgets.Checkbox(value=True, description='betti_PHT', indent=False),
        widgets.Checkbox(value=True, description='betti', indent=False),
        widgets.Checkbox(value=True, description='PI', indent=False),
        pi_local_checkbox
    ]

    root_path_text = widgets.Text(
        value='/base/path/to/put/the/output/folder/',
        placeholder='Base output path',
        description='Base output path:',
        disabled=False,
        layout=Layout(width='auto'),
        style={'description_width': '12%'}
    )

    vessel_path_text = widgets.Text(
        value='/path/to/the/segmentation/files/',
        placeholder='Enter vessel segmentation path',
        description='Segmentations Path:',
        disabled=False,
        layout=Layout(width='auto'),
        style={'description_width': '12%'}
    )



    tile_size_input = widgets.Text(
        value='8,8,8', 
        placeholder='e.g., 8,8,8',
        description='Tile Sizes (K,H,W):',
        disabled=False,
        layout=Layout(width='25%'),
        style={'description_width': '70%'}
    )

    save_npy_checkbox = widgets.Checkbox(
        value=True,
        description='Save NPY files',
        disabled=False,
        indent=False
    )

    generate_images_checkbox = widgets.Checkbox(
        value=True,
        description='Generate Graphs',
        disabled=False,
        indent=False
    )

    plot_per_patch_checkbox = widgets.Checkbox(
        value=True,
        description='Plot Per Patch (PI_local)',
        disabled=False,
        indent=False
    )

    def update_plot_per_patch_state(change):
        plot_per_patch_checkbox.disabled = not change.new
        if not change.new:
            plot_per_patch_checkbox.value = False
        tile_size_input.disabled = not change.new 

    pi_local_checkbox.observe(update_plot_per_patch_state, names='value')

    
    def observe_pi_checkbox(change):
        if not change.new: 
            tile_size_input.disabled = True
            
            if pi_local_checkbox.value: 
                pi_local_checkbox.value = False 
            pi_local_checkbox.disabled = True 
        else:
            pi_local_checkbox.disabled = False 
           
            tile_size_input.disabled = not pi_local_checkbox.value 



    pi_checkbox.observe(observe_pi_checkbox, names='value')


 
    if not pi_checkbox.value:
        tile_size_input.disabled = True
        tile_size_input.value = '0,0,0'
        pi_local_checkbox.disabled = True
        pi_local_checkbox.value = False 
    else:
        
        tile_size_input.disabled = not pi_local_checkbox.value

    plot_per_patch_checkbox.disabled = not pi_local_checkbox.value
    if not pi_local_checkbox.value:
        plot_per_patch_checkbox.value = False

    def observe_save_npy(change):

        if not change.new and not generate_images_checkbox.value:
            save_npy_checkbox.value = True



    def observe_generate_images(change):

        if not change.new and not save_npy_checkbox.value:
            generate_images_checkbox.value = True
    


    save_npy_checkbox.observe(observe_save_npy, names='value')
    generate_images_checkbox.observe(observe_generate_images, names='value')


    run_button = widgets.Button(
        description='Run Analysis',
        disabled=False,
        button_style='success',
        tooltip='Click to run the topological feature extraction',
        icon='play'
    )

    output_console = widgets.Output()

    def on_run_button_clicked(b):
        with output_console:
            output_console.clear_output()

            selected_features = [cb.description for cb in feature_names_checkboxes if cb.value]

            root_val = root_path_text.value
            vessel_val = vessel_path_text.value
            save_npy_val = save_npy_checkbox.value
            generate_images_val = generate_images_checkbox.value
            plot_per_patch_val = plot_per_patch_checkbox.value
            
  
            tile_size_input_str = tile_size_input.value
            try:
                tile_size_val = tuple(map(int, tile_size_input_str.split(',')))
                if len(tile_size_val) != 3:
                    raise ValueError("Expected 3 comma-separated values.")
            except ValueError:
                print(f"Invalid format for 'Tile Sizes'. Please enter three comma-separated integers (e.g., 8,8,8). Defaulting to (8,8,8).")
                tile_size_val = (8,8,8)
                tile_size_input.value = '8,8,8'
            
            # Validate individual tile sizes
            if 'PI_local' in selected_features and any(ts <= 1 for ts in tile_size_val):
                print(f"Invalid value(s) for 'Tile Sizes' {tile_size_val}. All values must be greater than 1. Defaulting to (8,8,8).")
                tile_size_val = (8,8,8)
                tile_size_input.value = '8,8,8' 
            
            print("\n" + "="*50 + "\n")
            try:
                extract_vessel_features(
                    featurenames=selected_features,
                    root=root_val,
                    vesselpath=vessel_val,
                    save_npy=save_npy_val,
                    generate_images=generate_images_val,
                    plot_per_patch=plot_per_patch_val,
                    tile_size=tile_size_val 
                )
                
                print("Topological feature extraction completed successfully!")
            except Exception as e:
                print(f"\n" + "="*50 + "\n")
                print(f"An error occurred: {e}")

    run_button.on_click(on_run_button_clicked)

    feature_box = widgets.VBox([widgets.Label("Select Features:")] + feature_names_checkboxes)

    input_paths_box = widgets.VBox([
        root_path_text,
        vessel_path_text,
        tile_size_input
    ])

    options_box = widgets.VBox([
        save_npy_checkbox,
        generate_images_checkbox,
        plot_per_patch_checkbox
    ])

    ui = widgets.VBox([
        widgets.HTML("<h2>Topological Feature Extraction Parameters</h2>"),
        feature_box,
        input_paths_box,
        options_box,
        widgets.HBox([run_button]),
        output_console
    ])

    display(ui)