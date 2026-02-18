import os
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from tqdm.notebook import tqdm
import bioio_tifffile
from ipyfilechooser import FileChooser
from skimage.morphology import dilation, disk, square
from scipy.ndimage import label
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
import SimpleITK as sitk
import logging
logging.getLogger("bioio").setLevel(logging.ERROR)


def get_surface_points(matrix):
    """
    Returns the coordinates (row, column) of the segmented pixels (True values).
    This is necessary for distance-based metrics like Hausdorff and MSD.
    """
    # np.argwhere returns the indices where the condition (matrix > 0) is met
    return np.argwhere(matrix.astype(bool))

def mean_surface_distance_unidirectional(points_A, points_B):
    """
    Calculates the Mean Surface Distance (MSD) from set A to set B:
    1/|A| * sum_{a in A} min_{b in B} ||a - b||
    
    Uses KDTree for efficient nearest neighbor search.
    """
    if len(points_A) == 0 or len(points_B) == 0:
        return 0.0 # Assuming 0 distance if sets are empty
    
    # Use KDTree to efficiently find the nearest neighbor in B for every point in A
    tree_B = KDTree(points_B)
    distances, _ = tree_B.query(points_A)
    
    return np.mean(distances)

def hausdorff_computing(n_annotators, predictions_list, m_gt):
    """
    Calculates the Symmetrical Hausdorff Distance (HD) between each prediction 
    and the Ground Truth (m_gt), and its generalized versions.
    The generalized metrics are the average of the pairwise distances.
    
    Arguments:
        n_annotators (int): Number of annotators to consider.
        predictions_list (list): List of numpy arrays (M_1, M_2, ...) binary (0/1).
        m_gt (np.ndarray): Numpy array of the ground truth (0/1).

    Returns:
        list: [HD(M_1, M_gt), HD(M_2, M_gt), ..., HD(M_n, M_gt), 
               HD_generalized_annotators, HD_generalized_model].
    """
    
    points_gt = get_surface_points(m_gt)
    hausdorff_results = []
    
    # Pairwise Comparison: HD(Mi, M_gt)
    for m_pred in predictions_list:
        points_pred = get_surface_points(m_pred)
        
        if len(points_pred) == 0 and len(points_gt) == 0:
             hd_sym = 0.0 
        elif len(points_pred) == 0 or len(points_gt) == 0:
             # Max distance, or 0.0 for simplified handling of empty sets
             hd_sym = 100
        else:
            # Symmetrical HD: max(h(A, B), h(B, A))
            h_ab = directed_hausdorff(points_pred, points_gt)[0]
            h_ba = directed_hausdorff(points_gt, points_pred)[0]
            hd_sym = max(h_ab, h_ba)
        
        hausdorff_results.append(hd_sym)

    n_to_consider = min(n_annotators, len(predictions_list))

    if n_to_consider > 0:
        subset_points = [get_surface_points(m) for m in predictions_list[:n_to_consider]]
        pair_distances = []
        
        #  Average HD among all pairs (M_i, M_j)
        for i in range(n_to_consider):
            for j in range(i + 1, n_to_consider):
                A = subset_points[i]
                B = subset_points[j]
                
                if len(A) == 0 and len(B) == 0:
                    pair_distances.append(0.0)
                elif len(A) == 0 or len(B) == 0:
                    pair_distances.append(0.0) 
                else:
                    h_ab = directed_hausdorff(A, B)[0]
                    h_ba = directed_hausdorff(B, A)[0]
                    pair_distances.append(max(h_ab, h_ba))
        
        # Report the average distance of all pairs
        hd_generalized_annotators = np.mean(pair_distances) if pair_distances else 0.0
        
        # Generalized Model: Average HD of each annotator Mi to M_gt
        hd_generalized_model = np.mean(hausdorff_results) if hausdorff_results else 0.0

    else:
        hd_generalized_annotators = 0.0
        hd_generalized_model = 0.0
    
    hausdorff_results.append(hd_generalized_annotators)
    hausdorff_results.append(hd_generalized_model)
    
    return hausdorff_results

def msd_computing(n_annotators, predictions_list, m_gt):
    """
    Calculates the Average Symmetric Surface Distance (ASSD, a form of MSD) 
    between each prediction and the Ground Truth (m_gt), and its generalized versions.
    The generalized metrics are the average of the pairwise distances.
    
    Arguments:
        n_annotators (int): Number of annotators to consider.
        predictions_list (list): List of numpy arrays (M_1, M_2, ...) binary (0/1).
        m_gt (np.ndarray): Numpy array of the ground truth (0/1).

    Returns:
        list: [MSD(M_1, M_gt), MSD(M_2, M_gt), ..., MSD(M_n, M_gt), 
               MSD_generalized_annotators, MSD_generalized_model].
    """
    
    points_gt = get_surface_points(m_gt)
    msd_results = []
    
    # MSD(Mi, M_gt) (ASSD)
    for m_pred in predictions_list:
        points_pred = get_surface_points(m_pred)
        
        if len(points_pred) == 0 and len(points_gt) == 0:
            msd_sym = 0.0 
        elif len(points_pred) == 0 or len(points_gt) == 0:
            msd_sym = 100
        else:
            #  (D(A->B) + D(B->A)) / 2
            d_ab = mean_surface_distance_unidirectional(points_pred, points_gt)
            d_ba = mean_surface_distance_unidirectional(points_gt, points_pred)
            msd_sym = (d_ab + d_ba) / 2
        
        msd_results.append(msd_sym)

    n_to_consider = min(n_annotators, len(predictions_list))

    if n_to_consider > 0:
        subset_points = [get_surface_points(m) for m in predictions_list[:n_to_consider]]
        pair_distances = []
        
        #  Generalized Annotators: Average MSD among all pairs (M_i, M_j)
        for i in range(n_to_consider):
            for j in range(i + 1, n_to_consider):
                A = subset_points[i]
                B = subset_points[j]
                
                if len(A) == 0 and len(B) == 0:
                    pair_distances.append(0.0) 
                elif len(A) == 0 or len(B) == 0:
                    pair_distances.append(0.0) 
                else:
                    d_ab = mean_surface_distance_unidirectional(A, B)
                    d_ba = mean_surface_distance_unidirectional(B, A)
                    pair_distances.append((d_ab + d_ba) / 2)
        
        msd_generalized_annotators = np.mean(pair_distances) if pair_distances else 0.0
        
        # Average MSD of each annotator Mi to M_gt
        msd_generalized_model = np.mean(msd_results) if msd_results else 0.0

    else:
        msd_generalized_annotators = 0.0
        msd_generalized_model = 0.0
    
    msd_results.append(msd_generalized_annotators)
    msd_results.append(msd_generalized_model)
    
    return msd_results

def jaccard_computing(n_annotators,predictions_list, m_gt):
    """
    Calculates the Jaccard Index (IoU) between each prediction matrix 
    and the ground truth matrix (m_gt), and the Generalized Jaccard 
    (M_1 through M_5 and m_gt). It ensures all input 0/1 matrices are 
    treated as booleans for set operations.

    Arguments:
        predictions_list (list): List of numpy arrays (M_1, M_2, ..., M_n) 
                                 containing binary values (0/1).
        m_gt (np.ndarray): Numpy array of the ground truth (0/1).

    Returns:
        list: [J(M_1, M_gt), J(M_2, M_gt), ..., J(M_n, M_gt), J_generalized_annotators, J_generalized_model].
    """


    m_gt_bool = m_gt.astype(bool)
    jaccard_results = []

    for m_pred in predictions_list:
        m_pred_bool = m_pred.astype(bool)
        
        intersection = np.sum(m_pred_bool & m_gt_bool)
        union = np.sum(m_pred_bool | m_gt_bool)
        
        # Jaccard Calculation: |A & B| / |A | B|
        jaccard = intersection / union if union != 0 else 1.0 
        
        jaccard_results.append(jaccard)

    n_to_consider = min(n_annotators, len(predictions_list))

    if n_to_consider > 0:
        subset_matrices_bool = [m.astype(bool) for m in predictions_list[:n_to_consider]]

        annotators_intersection = np.sum(np.logical_and.reduce(subset_matrices_bool))
        annotators_union = np.sum(np.logical_or.reduce(subset_matrices_bool))
        jaccard_generalized_annotators = annotators_intersection / annotators_union \
                                if annotators_union != 0 else 1.0
        
        generalized_matrices = subset_matrices_bool + [m_gt_bool]
        generalized_intersection = np.sum(np.logical_and.reduce(generalized_matrices))
        generalized_union = np.sum(np.logical_or.reduce(generalized_matrices))
        jaccard_generalized_model = generalized_intersection / generalized_union \
                                if generalized_union != 0 else 1.0
    else:
        jaccard_generalized_annotators = 0.0
        jaccard_generalized_model = 0.0
    
    jaccard_results.append(jaccard_generalized_annotators)
    jaccard_results.append(jaccard_generalized_model)
    
    return jaccard_results

def binarizar_matrix(matrix):
    if len(matrix.shape) == 2:
        matrix = matrix[None,...]
    
    if len(matrix.shape) != 3:
        raise ValueError(f"Image dims {matrix.shape} but ZYX are required")
    
    binary = []
    for zz in range(matrix.shape[0]):     
        mask = matrix[zz] != 0
        matrix_bin = np.zeros_like(matrix[zz], dtype=np.uint8)
        matrix_bin[mask] = 1
        binary.append(matrix_bin.astype(np.uint8))


    return np.stack(binary, axis=0).astype(np.uint8)


def thicken_segmentation_skimage(binary_matrix, n_pixels, kernel_shape):
    if len(binary_matrix.shape) == 2:
        binary_matrix = binary_matrix[None,...]
    
    if len(binary_matrix.shape) != 3:
        raise ValueError(f"Image dims {binary_matrix.shape} but ZYX are required")    

    if kernel_shape == 'disk':
        selem = disk(n_pixels)
    elif kernel_shape == 'square':
        side_length = 2 * n_pixels + 1
        selem = square(side_length)
    else:
        raise ValueError("The kernel_shape must be 'square' or 'disk'.")
    
    thickened_segmentation = []
    for zz in range(binary_matrix.shape[0]):
        thickened_segmentation.append(dilation(binary_matrix[zz], footprint=selem).astype(np.uint8))

    return np.stack(thickened_segmentation, axis=0)

def read_file(fn):
    try:
        img_pred = BioImage(fn, reader=bioio_tifffile.Reader).get_image_data("ZYX", C=0, T=0)
    except Exception as e:
        try:
            img_pred = BioImage(fn).get_image_data("ZYX", C=0, T=0)  
        except Exception as e:
            raise ValueError("Error at reading time.")
            return
    
    return img_pred

def extract_sv(matrix, target_class):
    if len(matrix.shape) == 2:
        matrix = matrix[None,...]
    
    if len(matrix.shape) != 3:
        raise ValueError(f"Image dims {matrix.shape} but ZYX are required")
    
    return (matrix == target_class).astype(np.uint8)

def count_components(matrix):
    if len(matrix.shape) == 2:
        matrix = matrix[None,...]
    
    if len(matrix.shape) != 3:
        raise ValueError(f"Image dims {matrix.shape} but ZYX are required")

    labeled_array, num_objects = label(matrix)
    
    return num_objects

def combine_binary_masks(mask_list):
    union_mask = mask_list[0].copy()
    intersection_mask = mask_list[0].copy()
    for i in range(1, len(mask_list)):
        current_mask = mask_list[i]
        union_mask = np.logical_or(union_mask, current_mask)
        intersection_mask = np.logical_and(intersection_mask, current_mask)

    union_mask = union_mask.astype(np.uint8)
    intersection_mask = intersection_mask.astype(np.uint8)
    
    return union_mask, intersection_mask



def add_missing_files(files_prediction,anotators_files,selected_gt):
    for fn in tqdm(files_prediction, desc= "Looking for missing annotations"):
        stem_file = fn.stem.replace('_segPred','') 
        ext = fn.suffix
        for folder in anotators_files:
            tiff_file = str(stem_file)+'.tiff'
            tif_file = str(stem_file)+'.tif'
            if (selected_gt / folder / tiff_file).exists():
                ext = '.tiff'
            elif (selected_gt / folder / tif_file).exists():
                ext = '.tif'
            else:
                out_img_file = str(stem_file) + str(ext) 
                save_path = selected_gt / folder / out_img_file
                img_pred = read_file(fn)
                zero_pred = np.zeros(img_pred.shape)
                OmeTiffWriter.save(data=zero_pred, uri=save_path, dim_order="ZYX")


def evaluation_metrics(anotators_files,selected_predictions,files_prediction,output_path,selected_gt,save_mask,dilatation,n_pixels,kernel_shape, eval_class, staple_threshold=0.5):

    if save_mask: 
        (selected_predictions.parent / 'Generated_masks').mkdir(parents=True, exist_ok=True)
    
    headers = [folder.split('_')[-1] for folder in anotators_files]
    headers.extend(['Union','Intersection','Staple','Model'])
    headers.insert(0,'image_id')
    jacci_head = []
    for head in headers:
        if head != 'image_id' and head != 'Model':
            jacci_head.append(head+'_jaccard')
    jacci_head.append('annotators_agree_jaccard')
    jacci_head.append('annotators_model_agree_jaccard')     
    headers.extend(jacci_head) 

    eval_id = str(selected_predictions).split('_')[-1]
    n_annotators = len(anotators_files)
    
    image_id = []
    counts_model = []
    counts_folder = []
    counts_unions = []
    counts_intersections = []
    counts_staple = []
    all_jaccard_indices = []
    all_hausdorff_distances = []
    all_mean_surface_distance = [] 
    all_sensitivity = []
    all_specificity = []  

    for fn in tqdm(files_prediction, desc= "Evaluation process"):
        stem_file = fn.stem.replace('_segPred','')
        image_id.append(stem_file)
        if dilatation is None:
            model_pred = binarizar_matrix(extract_sv(read_file(fn), eval_class))   
        else:
            model_pred = thicken_segmentation_skimage(binarizar_matrix(extract_sv(read_file(fn), eval_class)),n_pixels,kernel_shape)      
        counts_model.append(count_components(model_pred)) 
        annotations = []
        count_folder = []
        for folder in anotators_files:
            tiff_file = str(stem_file)+'.tiff'
            tif_file = str(stem_file)+'.tif'
            try:
                ann_im = read_file(selected_gt / folder / tiff_file)
            except Exception as e:
                try:
                    ann_im = read_file(selected_gt / folder / tif_file) 
                except Exception as e:
                    raise ValueError(f"Error founding {stem_file} in {folder}.")
            if model_pred.shape != ann_im.shape:
                raise ValueError(f"Image {stem_file} has different sahaoe for {folder} model prediction {model_pred.shape} differ from {ann_im.shape}.") 
            if dilatation is None:
                ann_im = binarizar_matrix(ann_im)
            else:
                ann_im = thicken_segmentation_skimage(binarizar_matrix(ann_im),n_pixels,kernel_shape)

            count_folder.append(count_components(ann_im)) 
            annotations.append(ann_im)
        
        counts_folder.append(count_folder)
        if n_annotators >= 2:             
            union, intersection = combine_binary_masks(annotations)
            staple_filter = sitk.STAPLEImageFilter()
            staple_filter.SetMaximumIterations (100)
            consensus = staple_filter.Execute([sitk.GetImageFromArray(arr) for arr in annotations])
            staple_mask = sitk.GetArrayFromImage(consensus > staple_threshold).astype(np.uint8) 
            annotations.extend([union, intersection, staple_mask])
            all_sensitivity.append(list(staple_filter.GetSensitivity())) 
            all_specificity.append(list(staple_filter.GetSpecificity()))
            counts_unions.append(count_components(union)) 
            counts_intersections.append(count_components(intersection)) 
            counts_staple.append(count_components(staple_mask))    
        else:
            shapeM = annotations[0].shape
            annotations.extend([np.zeros(shapeM), np.zeros(shapeM), np.zeros(shapeM)])
            all_sensitivity.append([0]) 
            all_specificity.append([0])
            counts_unions.append(0) 
            counts_intersections.append(0) 
            counts_staple.append(0)    
        
        all_jaccard_indices.append(jaccard_computing(n_annotators,annotations, model_pred))
        all_hausdorff_distances.append(hausdorff_computing(n_annotators,annotations, model_pred))
        all_mean_surface_distance.append(msd_computing(n_annotators,annotations, model_pred))
            
        if save_mask and (n_annotators >= 2):
            OmeTiffWriter.save(data=union, uri=selected_predictions.parent / 'Generated_masks' /fn.name.replace('segPred','union'), dim_order="ZYX")
            OmeTiffWriter.save(data=intersection, uri=selected_predictions.parent / 'Generated_masks' /fn.name.replace('segPred','intersection'), dim_order="ZYX")
            OmeTiffWriter.save(data=staple_mask, uri=selected_predictions.parent / 'Generated_masks' /fn.name.replace('segPred','staple_mask'), dim_order="ZYX")

  
 
    
    transposed_counts_folder = list(zip(*counts_folder))
    data = {headers[0]: image_id} 

    annotator_headers = headers[1:len(anotators_files) + 1]
    for i, header in enumerate(annotator_headers):
        data[header] = list(transposed_counts_folder[i])

    data['Union'] = counts_unions
    data['Intersection'] = counts_intersections
    data['Staple'] = counts_staple
    data['Model'] = counts_model

    transposed_jaccard = list(zip(*all_jaccard_indices))
    for i, header in enumerate(jacci_head):
       data[header] = list(transposed_jaccard[i])

    df = pd.DataFrame(data)
    #compute dice using jaccard 2J/1+J
    jaccard_columns = [col for col in df.columns if col.endswith('_jaccard')]
    for j_col in jaccard_columns:
        dice_col = j_col.replace('_jaccard', '_dice')
        df[dice_col] = df[j_col].apply(lambda j: (2 * j) / (1 + j) if j >= 0 else 0)
    
    transposed_hausdorff = list(zip(*all_hausdorff_distances)) 
    for i, j_col in enumerate(jaccard_columns):
        hausdorff_col = j_col.replace('_jaccard', '_hausdorff')
        df[hausdorff_col] = list(transposed_hausdorff[i])

    transposed_mean_surface_distance = list(zip(*all_mean_surface_distance)) 
    for i, j_col in enumerate(jaccard_columns):
        mean_surface_distance_col = j_col.replace('_jaccard', '_msd')
        df[mean_surface_distance_col] = list(transposed_mean_surface_distance[i])
    
    transposed_sensitivity = list(zip(*all_sensitivity))
    for i ,col in enumerate(headers[1:n_annotators+1]):  
        sens_col = col+'_sensivity'
        df[sens_col] = list(transposed_sensitivity[i])
    
    transposed_specificity = list(zip(*all_specificity))
    for i ,col in enumerate(headers[1:n_annotators+1]):  
        spe_col = col+'_specificity'
        df[spe_col] = list(transposed_specificity[i])
    
    if n_annotators < 2:
        df = df.iloc[:, [0,1,5,6,12,18,24]]
    csv_filename = f'model_evaluation_{eval_id}.csv'
    csv_path = output_path / csv_filename
    df.to_csv(csv_path, index=False)


def plot_curves(dataframe, columns, title, filename, color_list_or_map,y_label):
    
    plt.figure(figsize=(12, 7))

    stats_text = ""
    if 'Agreement' in title:
        if len(columns) >= 2:
            
            col_annotators = columns[0] 
            col_annotators_model = columns[1]
            mean_ann = dataframe[col_annotators].mean()
            std_ann = dataframe[col_annotators].std()
            
            mean_ann_model = dataframe[col_annotators_model].mean()
            std_ann_model = dataframe[col_annotators_model].std()
            
            stats_text = (
                f"\n\n"
                f"mean_annotators: {mean_ann:.4f}\n"
                f"std_annotators: {std_ann:.4f}\n\n"
                f"mean_annotators_model: {mean_ann_model:.4f}\n"
                f"std_annotators_model: {std_ann_model:.4f}\n"
            )
    

    legend_handles = []
    legend_labels = []

    if isinstance(color_list_or_map, list):
        colors_to_use = color_list_or_map
    else:
        colors_to_use = [color_list_or_map(i) for i in range(len(columns))]

    for i, col in enumerate(columns):
        color = colors_to_use[i % len(colors_to_use)]
        line, = plt.plot(dataframe.index, dataframe[col], label=col, color=color, linewidth=1.5)

    plt.title(f'{title}', fontsize=16)
    plt.xlabel('Images', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    legend_title = f"Columns{stats_text}"
    plt.legend(handles=legend_handles, labels=legend_labels, title=legend_title, 
               bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.grid(False) 
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_distributions(dataframe, columns, title, filename_base, y_label):

    data_to_plot = [dataframe[col].dropna().values for col in columns]
    column_labels = [col.replace('_jaccard', '').replace('_dice', '') for col in columns]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=column_labels, patch_artist=True)
    plt.title(title, fontsize=16)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(False)
            
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'{filename_base}_boxplot.png', dpi=300)
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.violinplot(data_to_plot, showmeans=True, showmedians=False, showextrema=True)
    
    plt.xticks(np.arange(1, len(column_labels) + 1), column_labels, ha='right')
    
    plt.title(title, fontsize=16)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(False)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'{filename_base}_violinplot.png', dpi=300)
    plt.close()

def graph_generator(output_path,selected_predictions,anotators_files):
    
    n_annotators = len(anotators_files)
    n_ann_jacc = n_annotators + 5
    

    eval_id = str(selected_predictions).split('_')[-1]
    (output_path / (eval_id +'_graphs')).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(output_path / f'model_evaluation_{eval_id}.csv')
    
    if n_annotators >= 2:
        counts_ann = df.columns[1:n_annotators].tolist() + ['Model']
        counts_gen = df.columns[n_annotators+1:2*n_annotators].tolist()

        jaccard_ann = df.columns[n_ann_jacc: n_ann_jacc+n_annotators].tolist()
        dice_ann = df.columns[n_ann_jacc+n_annotators+5:n_ann_jacc+2*n_annotators+5].tolist()

        jaccard_gen = df.columns[n_ann_jacc+n_annotators:n_ann_jacc+n_annotators+3].tolist()
        dice_gen = df.columns[n_ann_jacc+2*n_annotators+5:n_ann_jacc+2*n_annotators+8].tolist()

        jaccard_agree = df.columns[n_ann_jacc+n_annotators+3:n_ann_jacc+n_annotators+5].tolist()
        dice_agree = df.columns[n_ann_jacc+2*n_annotators+8:n_ann_jacc+2*n_annotators+10].tolist()

        hausdorff_ann = df.columns[n_ann_jacc+2*n_annotators+10:n_ann_jacc+2*n_annotators+n_annotators+10].tolist()
        msd_ann = df.columns[n_ann_jacc+2*n_annotators+n_annotators+15:n_ann_jacc+3*n_annotators+n_annotators+15].tolist()

        hausdorff_gen = df.columns[ n_ann_jacc+2*n_annotators+n_annotators+10:n_ann_jacc+2*n_annotators+n_annotators+13].tolist()
        msd_gen = df.columns[n_ann_jacc+3*n_annotators+n_annotators+15:n_ann_jacc+3*n_annotators+n_annotators+18].tolist()

        hausdorff_agree = df.columns[n_ann_jacc+2*n_annotators+n_annotators+13:n_ann_jacc+2*n_annotators+n_annotators+15].tolist()
        msd_agree = df.columns[n_ann_jacc+3*n_annotators+n_annotators+18:n_ann_jacc+3*n_annotators+n_annotators+20].tolist()

        sensitivity_ann = df.columns[n_ann_jacc+3*n_annotators+n_annotators+20:n_ann_jacc+4*n_annotators+n_annotators+20].tolist()
        specificity_ann = df.columns[n_ann_jacc+4*n_annotators+n_annotators+20:n_ann_jacc+5*n_annotators+n_annotators+20].tolist()
        

        colors = plt.cm.get_cmap('tab10', max(len(jaccard_ann), len(dice_ann)))


        plot_curves(df,counts_ann,'Annotators vs Model Findings', output_path / (eval_id +'_graphs') / 'annotators_model_findings.png', colors,'Findings')
        plot_curves(df,counts_gen,'Generated Masks vs Model Findings', output_path / (eval_id +'_graphs') / 'generated_masks_model_findings.png', colors,'Findings')
        
        plot_curves(df,jaccard_ann,'Annotators vs Model', output_path / (eval_id +'_graphs') / 'annotators_model_jaccard.png', colors,'Jaccard index')
        plot_curves(df,dice_ann,'Annotators vs Model', output_path / (eval_id +'_graphs') / 'annotators_model_dice.png', colors,'Dice index')
        plot_curves(df,hausdorff_ann,'Annotators vs Model', output_path / (eval_id +'_graphs') / 'annotators_model_hausdorff.png', colors,'Hausdorff distance')
        plot_curves(df,msd_ann,'Annotators vs Model', output_path / (eval_id +'_graphs') / 'annotators_model_msd.png', colors,'Mean surface distance')
        
        plot_curves(df,jaccard_gen,'Generated Mask vs Model', output_path / (eval_id +'_graphs') / 'generated_masks_model_jaccard.png', colors,'Jaccard index')
        plot_curves(df,dice_gen,'Generated Mask vs Model', output_path / (eval_id +'_graphs') / 'generated_masks_model_dice.png', colors,'Dice index')
        plot_curves(df,hausdorff_gen,'Generated Mask vs Model', output_path / (eval_id +'_graphs') / 'generated_masks_model_hausdorff.png', colors,'Hausdorff distance')
        plot_curves(df,msd_gen,'Generated Mask vs Model', output_path / (eval_id +'_graphs') / 'generated_masks_model_msd.png', colors,'Mean surface distance')

        plot_curves(df,jaccard_agree,'Jaccard Agreement', output_path / (eval_id +'_graphs') / 'agreement_jaccard.png', colors,'Jaccard index')
        plot_curves(df,dice_agree,'Dice Agreement', output_path / (eval_id +'_graphs') / 'agreement_dice.png', colors,'Dice index')
        plot_curves(df,hausdorff_agree,'Hausdorff Agreement', output_path / (eval_id +'_graphs') / 'agreement_hausdorff.png', colors,'Hausdorff distance')
        plot_curves(df,msd_agree,'Mean surface Agreement', output_path / (eval_id +'_graphs') / 'agreement_msd.png', colors,'Mean surface distance')


        plot_curves(df,sensitivity_ann,'STAPLE Sensitivity', output_path / (eval_id +'_graphs') / 'staple_sensitivity.png', colors,'Sensitivity')
        plot_curves(df,specificity_ann,'STAPLE Specificity', output_path / (eval_id +'_graphs') / 'staple_specificity.png', colors,'Specificity')
        
        plot_distributions(df, counts_ann, 'Annotators vs Model Findings', output_path / (eval_id +'_graphs') / 'annotators_model_findings', 'Findings')
        plot_distributions(df, counts_gen, 'Generated Masks vs Model Findings', output_path / (eval_id +'_graphs') / 'generated_masks_model_findings', 'Findings')

        plot_distributions(df, jaccard_ann, 'Annotators vs Model', output_path / (eval_id +'_graphs') / 'annotators_model_jaccard', 'Jaccard Index')
        plot_distributions(df, dice_ann, 'Annotators vs Model', output_path / (eval_id +'_graphs') / 'annotators_model_dice', 'Dice Index')
        plot_distributions(df, hausdorff_ann, 'Annotators vs Model', output_path / (eval_id +'_graphs') / 'annotators_model_hausdorff', 'Hausdorff distance')
        plot_distributions(df, msd_ann, 'Annotators vs Model', output_path / (eval_id +'_graphs') / 'annotators_model_msd', 'Mean surface distance')

    
        plot_distributions(df, jaccard_gen, 'Generated Mask vs Model', output_path / (eval_id +'_graphs') / 'generated_masks_model_jaccard', 'Jaccard Index')
        plot_distributions(df, dice_gen, 'Generated Mask vs Model', output_path / (eval_id +'_graphs') / 'generated_masks_model_dice', 'Dice Index')
        plot_distributions(df, hausdorff_gen, 'Generated Mask vs Model', output_path / (eval_id +'_graphs') / 'generated_masks_model_hausdorff', 'Hausdorff distance')
        plot_distributions(df, msd_gen, 'Generated Mask vs Model', output_path / (eval_id +'_graphs') / 'generated_masks_model_msd', 'Mean surface distance')

        plot_distributions(df, jaccard_agree, 'Jaccard Agreement', output_path / (eval_id +'_graphs') / 'agreement_jaccard', 'Jaccard Index')
        plot_distributions(df, dice_agree, 'Dice Agreement', output_path / (eval_id +'_graphs') / 'agreement_dice', 'Dice Index')
        plot_distributions(df, hausdorff_agree, 'Hausdorff Agreement', output_path / (eval_id +'_graphs') / 'agreement_hausdorff', 'Hausdorff distance')
        plot_distributions(df, msd_agree, 'Mean surface Agreement', output_path / (eval_id +'_graphs') / 'agreement_msd', 'Mean surface distance')

        plot_distributions(df, sensitivity_ann, 'STAPLE Sensitivity', output_path / (eval_id +'_graphs') / 'staple_sensitivity', 'Sensitivity')
        plot_distributions(df, specificity_ann, 'STAPLE Specificity', output_path / (eval_id +'_graphs') / 'staple_specificity', 'Specificity')
    else:
        counts_ann = df.columns[1:3].tolist()
        jaccard_ann = [df.columns[3]]
        dice_ann = [df.columns[4]]
        hausdorff_ann = [df.columns[5]]
        msd_ann = [df.columns[6]]

        colors = plt.cm.get_cmap('tab10', max(len(jaccard_ann), len(dice_ann)))

        plot_curves(df,counts_ann,'Annotator vs Model Findings', output_path / (eval_id +'_graphs') / 'annotator_model_findings.png', ['blue', 'green'],'Findings')
        plot_curves(df,jaccard_ann,'Annotator vs Model', output_path / (eval_id +'_graphs') / 'annotator_model_jaccard.png', colors,'Jaccard index')
        plot_curves(df,dice_ann,'Annotator vs Model', output_path / (eval_id +'_graphs') / 'annotator_model_dice.png', colors,'Dice index')
        plot_curves(df,hausdorff_ann,'Annotator vs Model', output_path / (eval_id +'_graphs') / 'annotator_model_hausdorff.png', colors,'Hausdorff distance')
        plot_curves(df,msd_ann,'Annotator vs Model', output_path / (eval_id +'_graphs') / 'annotator_model_msd.png', colors,'Mean surface distance')

        plot_distributions(df, counts_ann, 'Annotator vs Model Findings', output_path / (eval_id +'_graphs') / 'annotator_model_findings', 'Findings')
        plot_distributions(df, jaccard_ann, 'Annotator vs Model', output_path / (eval_id +'_graphs') / 'annotator_model_jaccard', 'Jaccard Index')
        plot_distributions(df, dice_ann, 'Annotator vs Model', output_path / (eval_id +'_graphs') / 'annotator_model_dice', 'Dice Index')
        plot_distributions(df, hausdorff_ann, 'Annotator vs Model', output_path / (eval_id +'_graphs') / 'annotator_model_hausdorff', 'Hausdorff distance')
        plot_distributions(df, msd_ann, 'Annotator vs Model', output_path / (eval_id +'_graphs') / 'annotator_model_msd', 'Mean surface distance')

 

def create_evaluation_menu():
    
    
    gt_path_widget = FileChooser(
        
        Path.cwd().as_posix(),
        title='Select grount truth folder',
        select_default=False 
    )
    gt_path_widget.show_only_dirs = True 
    gt_path_widget.layout = widgets.Layout(width='80%')

    predictions_path_widget = FileChooser(
        Path.cwd().as_posix(), 
        title='Select model predictions folder',
        select_default=False 
    )

    eval_class_widget = widgets.BoundedIntText(
        value=2,           
        min=0,             
        step=1,
        description='Evaluation Class:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )

    staple_threshold_widget = widgets.BoundedFloatText(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.01,
        description='STAPLE Threshold:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px', display='none') # Oculto por defecto
    )

    predictions_path_widget.show_only_dirs = True 
    predictions_path_widget.layout = widgets.Layout(width='80%')

    use_dilatation_checkbox = widgets.Checkbox(
        value=True,
        description="Use Dilatation",
        disabled=False
    )

    dilation_pixels_widget = widgets.BoundedIntText(
        value=3,
        min=1,
        step=1,
        description='Radius:',
        disabled=False,
        layout=widgets.Layout(width='150px')
    )

    dilation_mode_widget = widgets.Dropdown(
        options=['disk', 'square'],
        value='disk',
        description='Mode:',
        disabled=False,
        layout=widgets.Layout(width='200px')
    )

    dilation_options_box = widgets.HBox(
        [dilation_pixels_widget, dilation_mode_widget],
        layout=widgets.Layout(margin='0 0 0 20px') 
    )

    def on_dilatation_change(change):
        if change['new']:
            dilation_options_box.layout.display = 'flex'
        else:
            dilation_options_box.layout.display = 'none'

    def check_gt_subfolders(chooser):
        if chooser.selected:
            path = Path(chooser.selected)
            if path.exists():
                # Contamos cuántas subcarpetas (anotadores) hay
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                if len(subdirs) > 2:
                    staple_threshold_widget.layout.display = 'flex'
                else:
                    staple_threshold_widget.layout.display = 'none'


    use_dilatation_checkbox.observe(on_dilatation_change, names='value')
    dilation_options_box.layout.display = 'flex' if use_dilatation_checkbox.value else 'none'

    gt_path_widget.register_callback(check_gt_subfolders)

    save_mask_checkbox = widgets.Checkbox(
        value=False,
        description="Save generated masks",
        disabled=False
    )

    run_button = widgets.Button(description="Run Evaluation" , button_style='success', )
    output = widgets.Output()

    display(gt_path_widget, predictions_path_widget,eval_class_widget, staple_threshold_widget, 
            use_dilatation_checkbox, dilation_options_box, 
            save_mask_checkbox, run_button, output)
    
    def on_button_clicked(b):
        with output:
            output.clear_output()
            
            if not gt_path_widget.selected or not predictions_path_widget.selected:
                raise ValueError("Please select correct input folders")
                return

            selected_gt = Path(gt_path_widget.selected)
            base_prediction_path = Path(predictions_path_widget.selected)
            
            # ------------------------------------------------------------------
            # INPUT DETECTION LOGIC: Single vs Batch
            # ------------------------------------------------------------------
            # Check if the selected folder directly contains images (Scenario A)
            direct_images = sorted(base_prediction_path.glob("*.tiff")) + sorted(base_prediction_path.glob('*.tif'))
            
            folders_to_process = []
            
            if direct_images:
                # Scenario A: Single Model
                print(f"Single model detected at {base_prediction_path.name}")
                folders_to_process.append(base_prediction_path)
            else:
                # Scenario B: Check for subdirectories (Batch Mode)
                subdirs = [d for d in base_prediction_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if subdirs:
                    print(f"Batch mode detected. Found {len(subdirs)} subfolders in {base_prediction_path.name}:")
                    for d in subdirs:
                         print(f" - {d.name}")
                    folders_to_process = sorted(subdirs)
                else:
                    raise ValueError(f"No .tif/.tiff files or subfolders found in {base_prediction_path}.")
            
            # ------------------------------------------------------------------
            # SETUP COMMON PARAMETERS
            # ------------------------------------------------------------------
            eval_class = eval_class_widget.value
            save_mask = save_mask_checkbox.value
            staple_thresh = staple_threshold_widget.value
            
            anotators_files = [item.name for item in selected_gt.iterdir() if item.is_dir()]
            if len(anotators_files) == 0:
                raise ValueError(f"No ground truth folders found in {selected_gt}.")

            dilatation = None
            n_pixels = None
            kernel_shape = None
            if use_dilatation_checkbox.value :
                dilatation = True
                n_pixels = dilation_pixels_widget.value
                kernel_shape = dilation_mode_widget.value

            print(f"###################################### { len(anotators_files) } annotators folders found in GT.######################################")
            
            # ------------------------------------------------------------------
            # BATCH EXECUTION LOOP
            # ------------------------------------------------------------------
            for current_pred_folder in folders_to_process:
                output.clear_output(wait=True)
                print(f"\n=================================================================================")
                print(f"PROCESSING PREDICTION MODEL: {current_pred_folder.name}")
                print(f"=================================================================================")
                
                # Identify images for current folder
                files_prediction = sorted(current_pred_folder.glob("*.tiff"))
                files_prediction.extend(list(current_pred_folder.glob('*.tif')))
                
                if not files_prediction:
                    print(f"WARNING: No images found in {current_pred_folder.name}. Skipping.")
                    continue
                
                # Determine Output Path
                # If single mode, this is base_path.parent / Eval_Output.
                # If batch mode, this is base_path / Eval_Output (since current_pred_folder is inside base_path).
                # This ensures all outputs are grouped logically.
                output_path = current_pred_folder.parent / 'Evaluation_output'
                output_path.mkdir(parents=True, exist_ok=True)
                
                print(f"Found {len(files_prediction)} prediction files.")
                
                print('--- Looking for missing annotations ---')
                add_missing_files(files_prediction, anotators_files, selected_gt)
                
                print('--- Generating evaluation metrics ---')
                # Note: evaluation_metrics writes a CSV using current_pred_folder name as ID
                evaluation_metrics(
                    anotators_files,
                    current_pred_folder,
                    files_prediction,
                    output_path,
                    selected_gt,
                    save_mask,
                    dilatation,
                    n_pixels,
                    kernel_shape, 
                    eval_class, 
                    staple_thresh
                )
                
                print('--- Generating plots ---')
                graph_generator(output_path, current_pred_folder, anotators_files)
                
                print(f"Completed: {current_pred_folder.name}")

            print('\n###################################### All evaluations completed ############################################')
            print(f'Evaluation results saved at {base_prediction_path.parent if direct_images else base_prediction_path} / Evaluation_output')
            

    run_button.on_click(on_button_clicked)

def save_summary_plots(df, folder_path, is_single=True):

    if df.empty:
        return

    label_col = 'Statistic' if 'Statistic' in df.columns else 'id'
    
    if is_single and label_col == 'Statistic':
        plot_df = df[df[label_col] == 'mean']
    else:
        plot_df = df

    if len(df.columns) <= 5:

        m_jaccard = [c for c in df.columns if 'jaccard' in c.lower()]
        m_dice = [c for c in df.columns if 'dice' in c.lower()]
        m_hausdorff = [c for c in df.columns if 'hausdorff' in c.lower()]
        m_msd = [c for c in df.columns if 'msd' in c.lower()]

        metrics_list = []
        if m_jaccard: metrics_list.append(('Jaccard', m_jaccard[0], 'left'))
        if m_dice: metrics_list.append(('Dice', m_dice[0], 'left'))
        if m_hausdorff: metrics_list.append(('Hausdorff', m_hausdorff[0], 'right'))
        if m_msd: metrics_list.append(('MSD', m_msd[0], 'right'))

        plt.figure(figsize=(12, 7))
        ax1 = plt.gca()
        ax2 = ax1.twinx() 

        n_models = len(plot_df)
        n_metrics = len(metrics_list)
        x = np.arange(n_metrics)
        width = 0.8 / n_models 

        colors = plt.cm.Set1(np.linspace(0, 1, n_models))
        
        legend_handles = []

        for i, (idx, row) in enumerate(plot_df.iterrows()):
            model_name = row[label_col]
            model_color = colors[i]
            

            vals_left = []
            x_left = []
            vals_right = []
            x_right = []

            for m_idx, (m_label, col_name, side) in enumerate(metrics_list):
                pos = m_idx + (i - n_models/2 + 0.5) * width
                val = row[col_name]
                
                if side == 'left':
                    bar = ax1.bar(pos, val, width, color=model_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                else:
                    bar = ax2.bar(pos, val, width, color=model_color, alpha=0.6, hatch='//', edgecolor='black', linewidth=0.5)
                
                if m_idx == 0:
                    legend_handles.append(plt.Rectangle((0,0),1,1, color=model_color, label=model_name))

        ax1.set_ylabel('Precision Scores (Jaccard / Dice)↑', color='blue', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2.set_ylabel('Distance Metrics (Hausdorff / MSD)↓', color='red', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('Model Comparison ', fontsize=14, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m[0] for m in metrics_list], fontsize=11)
        ax1.set_xlabel('Evaluation Metrics', fontsize=12)
        
        ax1.legend(handles=legend_handles, title="Models", loc='upper left', bbox_to_anchor=(1.15, 1))
        

        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        save_name = folder_path / "summary_plot.png"
        plt.savefig(save_name, dpi=300)
        plt.close()
        print(f"Generated dual-axis bar plot: {save_name}")

    else:
        metrics = {
            'Jaccard': '_jaccard',
            'Dice': '_dice',
            'MSD': '_msd',
            'Hausdorff': '_hausdorff'
        }

        label_col = 'Statistic' if 'Statistic' in df.columns else 'id'
        
        if is_single:
            plot_df = df[df[label_col] == 'mean']
        else:
            plot_df = df

        for title, suffix in metrics.items():
            relevant_cols = [c for c in df.columns if c.endswith(suffix)]
            
            if not relevant_cols:
                continue

            plt.figure(figsize=(12, 7))
            
            x_labels = [c.replace(suffix, '') for c in relevant_cols]
            
            for _, row in plot_df.iterrows():
                label = row[label_col]
                values = row[relevant_cols].values
                plt.plot(x_labels, values, marker='o', label=label)

            if title == 'Dice' or title == 'Jaccard':
                titlef = f'Summary of {title} Metrics ↑' 
            else:
                titlef = f'Summary of {title} Metrics ↓' 


            plt.title(titlef)
            plt.xlabel('Annotator')
            plt.ylabel('Mean')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.6)
            if not is_single or len(plot_df) > 1:
                plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            save_name = folder_path / f"summary_plot_{title.lower()}.png"
            plt.savefig(save_name,dpi=300)
            plt.close()



def generate_statistical_summaries(folder_path: Path):
    """
    Processes CSV files in a given folder to generate statistical summaries.

    Mode of Operation:
    1. Single CSV: Generates one summary file ('single_summary.csv') 
       with mean, std, and variance as rows. Skips min/max calculation.
    2. Multiple CSVs: Generates four summary files: mean, std, variance 
       (one CSV each), and min/max IDs ('min_max_summary.csv'). 
       If all values for a column/metric are identical, the ID is set to 
       'all the same' in the min/max summary.

    Args:
        folder_path (Path): Path object pointing to the directory with the CSVs.
    """
    

    mean_data = []
    std_data = []
    var_data = []

    
    output_filenames = [
        'mean_summary.csv', 
        'std_summary.csv', 
        'variance_summary.csv', 
        'min_max_summary.csv',
        'single_summary.csv' 
    ]

  
    csv_files = list(folder_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No .csv files found in {folder_path}")
        return


    for file in tqdm(csv_files, desc= "Sumary process"):
        
        if file.name in output_filenames:
            continue

        try:
           
            file_id = file.stem.split('_')[-1]

            df = pd.read_csv(file)

            if 'Model' not in df.columns:
                print(f"Warning: File {file.name} does not have a 'Model' column. Skipping.")
                continue

            model_index = df.columns.get_loc('Model')
            target_cols = df.columns[model_index + 1:]
            
            filtered_df = df[target_cols]

            mean_series = filtered_df.mean(numeric_only=True)
            std_series = filtered_df.std(numeric_only=True)
            var_series = filtered_df.var(numeric_only=True)

            mean_dict = mean_series.to_dict()
            std_dict = std_series.to_dict()
            var_dict = var_series.to_dict()

            mean_dict['id'] = file_id
            std_dict['id'] = file_id
            var_dict['id'] = file_id

            mean_data.append(mean_dict)
            std_data.append(std_dict)
            var_data.append(var_dict)

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    processed_count = len(mean_data)
    
    if processed_count == 0:
        print("No valid input files were processed. Exiting.")
        return

    if processed_count == 1:        
        mean_result = mean_data[0]
        std_result = std_data[0]
        var_result = var_data[0]
        
        mean_result.pop('id', None) 
        std_result.pop('id', None)
        var_result.pop('id', None)
        
        mean_result['Statistic'] = 'mean'
        std_result['Statistic'] = 'std'
        var_result['Statistic'] = 'variance'
        
        single_summary_list = [mean_result, std_result, var_result]
        df_single = pd.DataFrame(single_summary_list)
        
        cols = ['Statistic'] + [c for c in df_single.columns if c != 'Statistic']
        df_single = df_single[cols]
        
        output_path = folder_path / 'single_summary.csv' 
        df_single.to_csv(output_path, index=False)
        save_summary_plots(df_single, folder_path, is_single=True)
        print(f"Generated single summary: {output_path}")
        return 

    def save_summary(data_list, filename):
        if not data_list:
            return None
        
        summary_df = pd.DataFrame(data_list)
        cols = ['id'] + [c for c in summary_df.columns if c != 'id']
        summary_df = summary_df[cols]
        
        output_path = folder_path / filename
        summary_df.to_csv(output_path, index=False)
        print(f"Generated: {output_path}")
        
        return summary_df

    df_mean_all = save_summary(mean_data, 'mean_summary.csv')
    if df_mean_all is not None:
        save_summary_plots(df_mean_all, folder_path, is_single=False)
    df_std_all = save_summary(std_data, 'std_summary.csv')
    df_var_all = save_summary(var_data, 'variance_summary.csv')


    min_max_rows = []

    def extract_min_max_ids(df, metric_name):
        """Helper to find IDs for min and max values in a dataframe, using 
           'all the same' if values are identical, or comma-separated list for ties."""
        if df is None or df.empty:
            return
        
        df_indexed = df.set_index('id')
        row_min = {'Statistic': f'min_{metric_name}'}
        row_max = {'Statistic': f'max_{metric_name}'}
        
        for col in df_indexed.columns:
            min_val = df_indexed[col].min()
            max_val = df_indexed[col].max()

            if min_val == max_val:
                row_min[col] = 'independent of model election'
                row_max[col] = 'independent of model election'  
            else:
                
                min_ids = df_indexed.index[df_indexed[col] == min_val].tolist()
                row_min[col] = ",".join(map(str, min_ids))

                max_ids = df_indexed.index[df_indexed[col] == max_val].tolist()
                row_max[col] = ",".join(map(str, max_ids))

        min_max_rows.append(row_min)
        min_max_rows.append(row_max)


    if df_mean_all is not None:
        extract_min_max_ids(df_mean_all, 'mean')
    
    if df_std_all is not None:
        extract_min_max_ids(df_std_all, 'std')

    if df_var_all is not None:
        extract_min_max_ids(df_var_all, 'variance')

    if min_max_rows:
        df_min_max = pd.DataFrame(min_max_rows)
        cols = ['Statistic'] + [c for c in df_min_max.columns if c != 'Statistic']
        df_min_max = df_min_max[cols]
        
        final_path = folder_path / 'min_max_summary.csv'
        df_min_max.to_csv(final_path, index=False)
        print(f"Generated: {final_path}")
    else:
        print("Could not generate min_max_summary.csv.")

def create_sumary_menu():
    
    
    path_widget = FileChooser(
        
        Path.cwd().as_posix(),
        title= "Select csv's file folder",
        select_default=False 
    )
    path_widget.layout = widgets.Layout(width='80%')

    run_button = widgets.Button(description="Run Sumary" , button_style='success', )
    output = widgets.Output()

    display(path_widget, run_button, output)
    
    def on_button_clicked(b):
        with output:
            output.clear_output()
            selected_path = path_widget.selected

            if not selected_path:
                raise ValueError("Please select correct csv's folders")
                return
            
            selected_path = Path(path_widget.selected)
            
            print('###################################### Sumary generation ############################################')
            generate_statistical_summaries(selected_path)
            print('###################################### Sumary complete ############################################')


            
            

    run_button.on_click(on_button_clicked)
