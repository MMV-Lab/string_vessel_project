import os
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
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
from scipy.ndimage import label, binary_erosion
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import logging
import torch
import monai
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
import yaml
from monai.transforms import Spacing, Resize
from monai.data import MetaTensor

logging.getLogger("bioio").setLevel(logging.ERROR)


def to_monai_tensor(matrix):
    return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def extract_boundary(mask, min_boundary_width=1, dilation_ratio=0.02):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
        
    dims = mask.shape
    diag = np.sqrt(sum(d**2 for d in dims))
    d = max(int(round(dilation_ratio * diag)), min_boundary_width)
    
    eroded = binary_erosion(mask, iterations=d)
    boundary = mask.astype(bool) & ~eroded
    return boundary


def boundary_iou(y_pred, y_true):
    bound_pred = extract_boundary(y_pred)
    bound_true = extract_boundary(y_true)
    
    intersection = np.sum(bound_pred & bound_true)
    union = np.sum(bound_pred | bound_true)
    
    if union == 0:
        return 1.0 if np.sum(bound_pred) == 0 and np.sum(bound_true) == 0 else 0.0
    return intersection / union


def overall_contour_agreement(y_pred, y_true):
    bound_pred = extract_boundary(y_pred)
    bound_true = extract_boundary(y_true)
    
    intersection = np.sum(bound_pred & bound_true)
    sum_bounds = np.sum(bound_pred) + np.sum(bound_true)
    
    if sum_bounds == 0:
        return 1.0 if np.sum(bound_pred) == 0 and np.sum(bound_true) == 0 else 0.0
    return 2.0 * intersection / sum_bounds


def hausdorff_computing(pr,n_annotators, predictions_list, m_gt):
    metric_fn = HausdorffDistanceMetric(percentile=pr, include_background=True, get_not_nans=False)
    m_gt_t = to_monai_tensor(m_gt)
    hausdorff_results = []
    
    for m_pred in predictions_list:
        m_pred_t = to_monai_tensor(m_pred)
        if m_pred_t.sum() == 0 and m_gt_t.sum() == 0:
            hd_sym = 0.0
        elif m_pred_t.sum() == 0 or m_gt_t.sum() == 0:
            hd_sym = 100.0
        else:
            metric_fn(y_pred=m_pred_t, y=m_gt_t)
            hd_sym = metric_fn.aggregate().item()
            metric_fn.reset()
        hausdorff_results.append(hd_sym)

    n_to_consider = min(n_annotators, len(predictions_list))

    if n_to_consider > 0:
        subset_tensors = [to_monai_tensor(m) for m in predictions_list[:n_to_consider]]
        pair_distances = []
        
        for i in range(n_to_consider):
            for j in range(i + 1, n_to_consider):
                A = subset_tensors[i]
                B = subset_tensors[j]
                
                if A.sum() == 0 and B.sum() == 0:
                    pair_distances.append(0.0)
                elif A.sum() == 0 or B.sum() == 0:
                    pair_distances.append(100.0) 
                else:
                    metric_fn(y_pred=A, y=B)
                    val = metric_fn.aggregate().item()
                    metric_fn.reset()
                    pair_distances.append(val)
        
        hd_generalized_annotators = np.mean(pair_distances) if pair_distances else 0.0
        hd_generalized_model = np.mean(hausdorff_results) if hausdorff_results else 0.0
    else:
        hd_generalized_annotators = 0.0
        hd_generalized_model = 0.0
    
    hausdorff_results.append(hd_generalized_annotators)
    hausdorff_results.append(hd_generalized_model)
    
    return hausdorff_results


def msd_computing(n_annotators, predictions_list, m_gt):
    metric_fn = SurfaceDistanceMetric(include_background=True, symmetric=True, get_not_nans=False)
    m_gt_t = to_monai_tensor(m_gt)
    msd_results = []
    
    for m_pred in predictions_list:
        m_pred_t = to_monai_tensor(m_pred)
        
        if m_pred_t.sum() == 0 and m_gt_t.sum() == 0:
            msd_sym = 0.0 
        elif m_pred_t.sum() == 0 or m_gt_t.sum() == 0:
            msd_sym = 100.0
        else:
            metric_fn(y_pred=m_pred_t, y=m_gt_t)
            msd_sym = metric_fn.aggregate().item()
            metric_fn.reset()
        
        msd_results.append(msd_sym)

    n_to_consider = min(n_annotators, len(predictions_list))

    if n_to_consider > 0:
        subset_tensors = [to_monai_tensor(m) for m in predictions_list[:n_to_consider]]
        pair_distances = []
        
        for i in range(n_to_consider):
            for j in range(i + 1, n_to_consider):
                A = subset_tensors[i]
                B = subset_tensors[j]
                
                if A.sum() == 0 and B.sum() == 0:
                    pair_distances.append(0.0) 
                elif A.sum() == 0 or B.sum() == 0:
                    pair_distances.append(100.0) 
                else:
                    metric_fn(y_pred=A, y=B)
                    val = metric_fn.aggregate().item()
                    metric_fn.reset()
                    pair_distances.append(val)
        
        msd_generalized_annotators = np.mean(pair_distances) if pair_distances else 0.0
        msd_generalized_model = np.mean(msd_results) if msd_results else 0.0
    else:
        msd_generalized_annotators = 0.0
        msd_generalized_model = 0.0
    
    msd_results.append(msd_generalized_annotators)
    msd_results.append(msd_generalized_model)
    
    return msd_results


def boundary_iou_computing(n_annotators, predictions_list, m_gt):
    biou_results = []
    
    for m_pred in predictions_list:
        biou_results.append(boundary_iou(m_pred, m_gt))

    n_to_consider = min(n_annotators, len(predictions_list))

    if n_to_consider > 0:
        pair_distances = []
        for i in range(n_to_consider):
            for j in range(i + 1, n_to_consider):
                pair_distances.append(boundary_iou(predictions_list[i], predictions_list[j]))
        
        generalized_annotators = np.mean(pair_distances) if pair_distances else 0.0
        generalized_model = np.mean(biou_results) if biou_results else 0.0
    else:
        generalized_annotators = 0.0
        generalized_model = 0.0
    
    biou_results.append(generalized_annotators)
    biou_results.append(generalized_model)
    
    return biou_results


def oca_computing(n_annotators, predictions_list, m_gt):
    oca_results = []
    
    for m_pred in predictions_list:
        oca_results.append(overall_contour_agreement(m_pred, m_gt))

    n_to_consider = min(n_annotators, len(predictions_list))

    if n_to_consider > 0:
        pair_distances = []
        for i in range(n_to_consider):
            for j in range(i + 1, n_to_consider):
                pair_distances.append(overall_contour_agreement(predictions_list[i], predictions_list[j]))
        
        generalized_annotators = np.mean(pair_distances) if pair_distances else 0.0
        generalized_model = np.mean(oca_results) if oca_results else 0.0
    else:
        generalized_annotators = 0.0
        generalized_model = 0.0
    
    oca_results.append(generalized_annotators)
    oca_results.append(generalized_model)
    
    return oca_results


def jaccard_computing(n_annotators,predictions_list, m_gt):
    m_gt_bool = m_gt.astype(bool)
    jaccard_results = []

    for m_pred in predictions_list:
        m_pred_bool = m_pred.astype(bool)
        
        intersection = np.sum(m_pred_bool & m_gt_bool)
        union = np.sum(m_pred_bool | m_gt_bool)
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

def extract_class(matrix, target_class):
    if len(matrix.shape) == 2:
        matrix = matrix[None,...]
    
    if len(matrix.shape) != 3:
        raise ValueError(f"Image dims {matrix.shape} but ZYX are required")
    
    if isinstance(target_class, int):
        class_f = (matrix == target_class).astype(np.uint8)
    elif isinstance(target_class, str):
        class_f = (matrix > 0).astype(np.uint8)
    else:
        raise ValueError(f"Check the target class str/int input is required but {target_class}-{type(target_class)} is given.")         
    
    return class_f

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


def evaluation_metrics(
    anotators_files,
    selected_predictions,
    files_prediction,
    output_path,
    selected_gt,
    save_mask,
    dilatation,
    n_pixels,
    kernel_shape,
    eval_class, 
    staple_threshold,
    eval_annotators,
    use_staple,
    use_union,
    use_inter
    ):

    if save_mask: 
        (selected_predictions.parent / 'Generated_masks').mkdir(parents=True, exist_ok=True)
    
    ann_headers = [folder.split('_')[-1] for folder in eval_annotators]
    gen_headers = []
    if use_union: gen_headers.append('Union')
    if use_inter: gen_headers.append('Intersection')
    if use_staple: gen_headers.append('Staple')
    
    all_eval_headers = ann_headers + gen_headers
    headers = ['image_id'] + all_eval_headers + ['Model']

    jacci_head = [h + '_jaccard' for h in all_eval_headers]
    jacci_head.append('annotators_agree_jaccard')
    jacci_head.append('annotators_model_agree_jaccard')

    eval_id = str(selected_predictions).split('_')[-1]
    n_primary = len(eval_annotators)
    
    image_id = []
    counts_model = []
    counts_folder = []
    all_jaccard_indices = []
    all_hausdorff_distances = []
    all_mean_surface_distance = [] 
    all_biou = []
    all_oca = []
    all_sensitivity = []
    all_specificity = []  

    for fn in tqdm(files_prediction, desc= "Evaluation process"):
        stem_file = fn.stem.replace('_segPred','')
        image_id.append(stem_file)

        if dilatation is None:
            model_pred = binarizar_matrix(extract_class(read_file(fn), eval_class))   
        else:
            model_pred = thicken_segmentation_skimage(binarizar_matrix(extract_class(read_file(fn), eval_class)),n_pixels,kernel_shape)      
        counts_model.append(count_components(model_pred)) 
        
        source_folders = eval_annotators if n_primary >= 2 else anotators_files
        folders_to_read = list(set(source_folders + eval_annotators))
        mask_cache = {}
        
        for folder in folders_to_read:
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
                raise ValueError(f"Image {stem_file} has different shape for {folder} model prediction {model_pred.shape} differ from {ann_im.shape}.")

            ann_im = binarizar_matrix(ann_im)
            if dilatation is not None:
                ann_im = thicken_segmentation_skimage(binarizar_matrix(ann_im),n_pixels,kernel_shape)
            mask_cache[folder] = ann_im
            
        current_eval_masks = [mask_cache[f] for f in eval_annotators]

        if use_union or use_inter or use_staple:
            source_masks = [mask_cache[f] for f in source_folders]
            if len(source_masks) >= 2:
                union, intersection = combine_binary_masks(source_masks)
                staple_filter = sitk.STAPLEImageFilter()
                staple_filter.SetMaximumIterations (100)
                consensus = staple_filter.Execute([sitk.GetImageFromArray(arr) for arr in source_masks])
                staple_mask = sitk.GetArrayFromImage(consensus > staple_threshold).astype(np.uint8) 
                
                if use_union: current_eval_masks.append(union)
                if use_inter: current_eval_masks.append(intersection)
                if use_staple: 
                    current_eval_masks.append(staple_mask)
                    if n_primary >= 2:
                        all_sensitivity.append(list(staple_filter.GetSensitivity())) 
                        all_specificity.append(list(staple_filter.GetSpecificity()))
                        
                if save_mask and (len(anotators_files) >= 2):
                    if use_union: OmeTiffWriter.save(data=union, uri=selected_predictions.parent / 'Generated_masks' /fn.name.replace('segPred','union'), dim_order="ZYX")
                    if use_inter: OmeTiffWriter.save(data=intersection, uri=selected_predictions.parent / 'Generated_masks' /fn.name.replace('segPred','intersection'), dim_order="ZYX")
                    if use_staple: OmeTiffWriter.save(data=staple_mask, uri=selected_predictions.parent / 'Generated_masks' /fn.name.replace('segPred','staple_mask'), dim_order="ZYX")
            else:
                blank_m = np.zeros_like(model_pred)
                if use_union: current_eval_masks.append(blank_m)
                if use_inter: current_eval_masks.append(blank_m)
                if use_staple: current_eval_masks.append(blank_m)
                if n_primary >= 2 and use_staple:
                    all_sensitivity.append([0]*n_primary) 
                    all_specificity.append([0]*n_primary)
                    
        counts_folder.append([count_components(m) for m in current_eval_masks])
        
        all_jaccard_indices.append(jaccard_computing(n_primary, current_eval_masks, model_pred))
        all_hausdorff_distances.append(hausdorff_computing(95, n_primary, current_eval_masks, model_pred))
        all_mean_surface_distance.append(msd_computing(n_primary, current_eval_masks, model_pred))
        all_biou.append(boundary_iou_computing(n_primary, current_eval_masks, model_pred))
        all_oca.append(oca_computing(n_primary, current_eval_masks, model_pred))

    data = {'image_id': image_id} 
    if counts_folder:
        transposed_counts = list(zip(*counts_folder))
        for i, h in enumerate(all_eval_headers):
            data[h] = list(transposed_counts[i])
            
    data['Model'] = counts_model

    transposed_jaccard = list(zip(*all_jaccard_indices))
    for i, h in enumerate(jacci_head):
       data[h] = list(transposed_jaccard[i])

    df = pd.DataFrame(data)
    
    jaccard_columns = [col for col in df.columns if col.endswith('_jaccard')]
    for j_col in jaccard_columns:
        dice_col = j_col.replace('_jaccard', '_dice')
        df[dice_col] = df[j_col].apply(lambda j: (2 * j) / (1 + j) if j >= 0 else 0)
    
    transposed_hausdorff = list(zip(*all_hausdorff_distances)) 
    transposed_mean_surface_distance = list(zip(*all_mean_surface_distance)) 
    transposed_biou = list(zip(*all_biou)) 
    transposed_oca = list(zip(*all_oca)) 
    
    for i, j_col in enumerate(jacci_head):
        df[j_col.replace('_jaccard', '_hausdorff')] = list(transposed_hausdorff[i])
        df[j_col.replace('_jaccard', '_msd')] = list(transposed_mean_surface_distance[i])
        df[j_col.replace('_jaccard', '_biou')] = list(transposed_biou[i])
        df[j_col.replace('_jaccard', '_oca')] = list(transposed_oca[i])
    
    if n_primary >= 2 and use_staple:
        transposed_sensitivity = list(zip(*all_sensitivity))
        transposed_specificity = list(zip(*all_specificity))
        for i, h in enumerate(ann_headers):  
            df[f"{h}_sensivity"] = list(transposed_sensitivity[i])
            df[f"{h}_specificity"] = list(transposed_specificity[i])
    
    if n_primary < 2:
        cols_to_drop = []
        for m in ['_jaccard', '_dice', '_hausdorff', '_msd', '_biou', '_oca']:
            cols_to_drop.extend([f"annotators_agree{m}", f"annotators_model_agree{m}"])
        df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        
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
    eval_id = str(selected_predictions).split('_')[-1]
    graphs_dir = output_path / (eval_id +'_graphs')
    graphs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(output_path / f'model_evaluation_{eval_id}.csv')
    
    base_cols = df.columns.tolist()
    ignore_cols = ['image_id', 'Model']
    metric_suffixes = ['_jaccard', '_dice', '_hausdorff', '_msd', '_biou', '_oca', '_sensivity', '_specificity']
    
    gt_names = [c for c in base_cols if c not in ignore_cols and not any(c.endswith(s) for s in metric_suffixes)]
    ann_names = [c for c in gt_names if c not in ['Union', 'Intersection', 'Staple']]
    gen_names = [c for c in gt_names if c in ['Union', 'Intersection', 'Staple']]
    
    has_agreement = 'annotators_agree_jaccard' in base_cols

    if len(ann_names) >= 2 or has_agreement:
        counts_ann = ann_names + ['Model'] if ann_names else ['Model']
        counts_gen = gen_names

        def get_metric_cols(suffix):
            ann = [f"{a}{suffix}" for a in ann_names if f"{a}{suffix}" in base_cols]
            gen = [f"{g}{suffix}" for g in gen_names if f"{g}{suffix}" in base_cols]
            agr = [f"annotators_agree{suffix}", f"annotators_model_agree{suffix}"] if has_agreement else []
            return ann, gen, agr

        metrics_map = [
            ('_jaccard', 'Jaccard index', 'jaccard'),
            ('_dice', 'Dice index', 'dice'),
            ('_hausdorff', 'Hausdorff distance', 'hausdorff'),
            ('_msd', 'Mean surface distance', 'msd'),
            ('_biou', 'Boundary IoU', 'biou'),
            ('_oca', 'Overall Contour Agreement', 'oca')
        ]

        colors = plt.cm.get_cmap('tab10', max(len(ann_names), 5))

        if ann_names:
            plot_curves(df, counts_ann, 'Annotators vs Model Findings', graphs_dir / 'annotators_model_findings.png', colors, 'Findings')
            plot_distributions(df, counts_ann, 'Annotators vs Model Findings', graphs_dir / 'annotators_model_findings', 'Findings')
            for suffix, title, m_name in metrics_map:
                ann_cols, _, _ = get_metric_cols(suffix)
                if ann_cols:
                    plot_curves(df, ann_cols, 'Annotators vs Model', graphs_dir / f'annotators_model_{m_name}.png', colors, title)
                    plot_distributions(df, ann_cols, 'Annotators vs Model', graphs_dir / f'annotators_model_{m_name}', title)

        if gen_names:
            plot_curves(df, gen_names, 'Generated Masks vs Model Findings', graphs_dir / 'generated_masks_model_findings.png', colors, 'Findings')
            plot_distributions(df, gen_names, 'Generated Masks vs Model Findings', graphs_dir / 'generated_masks_model_findings', 'Findings')
            for suffix, title, m_name in metrics_map:
                _, gen_cols, _ = get_metric_cols(suffix)
                if gen_cols:
                    plot_curves(df, gen_cols, 'Generated Mask vs Model', graphs_dir / f'generated_masks_model_{m_name}.png', colors, title)
                    plot_distributions(df, gen_cols, 'Generated Mask vs Model', graphs_dir / f'generated_masks_model_{m_name}', title)

        if has_agreement:
            for suffix, title, m_name in metrics_map:
                _, _, agr_cols = get_metric_cols(suffix)
                if agr_cols:
                    plot_curves(df, agr_cols, f'{title} Agreement', graphs_dir / f'agreement_{m_name}.png', colors, title)
                    plot_distributions(df, agr_cols, f'{title} Agreement', graphs_dir / f'agreement_{m_name}', title)

        if 'Staple' in gen_names and ann_names:
            sens_cols = [f"{a}_sensivity" for a in ann_names if f"{a}_sensivity" in base_cols]
            spec_cols = [f"{a}_specificity" for a in ann_names if f"{a}_specificity" in base_cols]
            if sens_cols:
                plot_curves(df, sens_cols, 'STAPLE Sensitivity', graphs_dir / 'staple_sensitivity.png', colors, 'Sensitivity')
                plot_distributions(df, sens_cols, 'STAPLE Sensitivity', graphs_dir / 'staple_sensitivity', 'Sensitivity')
            if spec_cols:
                plot_curves(df, spec_cols, 'STAPLE Specificity', graphs_dir / 'staple_specificity.png', colors, 'Specificity')
                plot_distributions(df, spec_cols, 'STAPLE Specificity', graphs_dir / 'staple_specificity', 'Specificity')
    else:
        target = gt_names[0] if gt_names else None
        if not target: return
        
        counts_single = [target, 'Model']
        colors = ['blue', 'green']
        
        plot_curves(df, counts_single, 'Annotator vs Model Findings', graphs_dir / 'annotator_model_findings.png', colors, 'Findings')
        plot_distributions(df, counts_single, 'Annotator vs Model Findings', graphs_dir / 'annotator_model_findings', 'Findings')
        
        metrics_map = [
            ('_jaccard', 'Jaccard index', 'jaccard'),
            ('_dice', 'Dice index', 'dice'),
            ('_hausdorff', 'Hausdorff distance', 'hausdorff'),
            ('_msd', 'Mean surface distance', 'msd'),
            ('_biou', 'Boundary IoU', 'biou'),
            ('_oca', 'Overall Contour Agreement', 'oca')
        ]
        
        for suffix, title, m_name in metrics_map:
            metric_col = f"{target}{suffix}"
            if metric_col in base_cols:
                plot_curves(df, [metric_col], 'Annotator vs Model', graphs_dir / f'annotator_model_{m_name}.png', colors, title)
                plot_distributions(df, [metric_col], 'Annotator vs Model', graphs_dir / f'annotator_model_{m_name}', title)


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

    outName = widgets.Text(
        value='Evaluation_output',
        placeholder='Folder name',
        description='Output folder name (optional):',
        disabled=False,
        style={'description_width':'initial'},
        layout=widgets.Layout(width='30%')
    )
 
    staple_threshold_widget = widgets.BoundedFloatText(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.01,
        description='STAPLE Threshold:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px', display='none') 
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

    gt_config_box = widgets.VBox()

    def check_gt_subfolders(chooser):
        if chooser.selected:
            path = Path(chooser.selected)
            if path.exists():
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                if len(subdirs) >= 2:
                    staple_threshold_widget.layout.display = 'flex'
                    save_mask_checkbox.layout.display = 'flex'
                    
                    title = widgets.HTML("<b>-- GT configurations --</b>")
                    all_ann_cb = widgets.Checkbox(value=True, description='all annotators')
                    indiv_cbs = [widgets.Checkbox(value=False, description=d.name) for d in subdirs]
                    staple_cb = widgets.Checkbox(value=False, description='STAPLE')
                    union_cb = widgets.Checkbox(value=False, description='Union')
                    inter_cb = widgets.Checkbox(value=False, description='Intersection')
                    
                    # Cambio a HBox con flex_flow para alinear horizontalmente y permitir saltos de línea
                    indiv_box = widgets.HBox(indiv_cbs + [staple_cb, union_cb, inter_cb])
                    indiv_box.layout.flex_flow = 'row wrap'
                    indiv_box.layout.display = 'none'
                    
                    def on_all_ann_change(change):
                        indiv_box.layout.display = 'none' if change['new'] else 'flex'
                    all_ann_cb.observe(on_all_ann_change, names='value')
                    
                    gt_config_box.children = [title, all_ann_cb, indiv_box]
                    
                    run_button.all_ann_cb = all_ann_cb
                    run_button.indiv_cbs = indiv_cbs
                    run_button.staple_cb = staple_cb
                    run_button.union_cb = union_cb
                    run_button.inter_cb = inter_cb
                    run_button.has_multi_gt = True
                else:
                    staple_threshold_widget.layout.display = 'none'
                    save_mask_checkbox.value = False
                    save_mask_checkbox.layout.display = 'none'
                    gt_config_box.children = []
                    run_button.has_multi_gt = False
    
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

    display(gt_path_widget, predictions_path_widget, outName, gt_config_box, staple_threshold_widget, 
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
            
            direct_images = sorted(base_prediction_path.glob("*.tiff")) + sorted(base_prediction_path.glob('*.tif'))
            
            folders_to_process = []
            
            if direct_images:
                print(f"Single model detected at {base_prediction_path.name}")
                folders_to_process.append(base_prediction_path)
            else:
                subdirs = [d for d in base_prediction_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
                if subdirs:
                    print(f"Batch mode detected. Found {len(subdirs)} subfolders in {base_prediction_path.name}:")
                    for d in subdirs:
                         print(f" - {d.name}")
                    folders_to_process = sorted(subdirs)
                else:
                    raise ValueError(f"No .tif/.tiff files or subfolders found in {base_prediction_path}.")

            eval_class = 2
            save_mask = save_mask_checkbox.value
            staple_thresh = staple_threshold_widget.value
            folder_name = outName.value
            anotators_files = [item.name for item in selected_gt.iterdir() if item.is_dir()]
            
            if len(anotators_files) == 0:
                raise ValueError(f"No ground truth folders found in {selected_gt}.")

            eval_annotators = []
            use_staple = use_union = use_intersection = False
            
            if getattr(run_button, 'has_multi_gt', False):
                if run_button.all_ann_cb.value:
                    eval_annotators = anotators_files
                    use_staple = use_union = use_intersection = True
                else:
                    eval_annotators = [cb.description for cb in run_button.indiv_cbs if cb.value]
                    use_staple = run_button.staple_cb.value
                    use_union = run_button.union_cb.value
                    use_intersection = run_button.inter_cb.value
                    
                if not eval_annotators and not (use_staple or use_union or use_intersection):
                    raise ValueError("Please select at least one Ground Truth configuration.")
            else:
                eval_annotators = anotators_files

            dilatation = None
            n_pixels = None
            kernel_shape = None
            if use_dilatation_checkbox.value :
                dilatation = True
                n_pixels = dilation_pixels_widget.value
                kernel_shape = dilation_mode_widget.value

            print(f"###################################### GT Configuration Settled. ######################################")
            
            for current_pred_folder in folders_to_process:
                output.clear_output(wait=True)
                print(f"\n=================================================================================")
                print(f"PROCESSING PREDICTION MODEL: {current_pred_folder.name}")
                print(f"=================================================================================")
                
                files_prediction = sorted(current_pred_folder.glob("*.tiff"))
                files_prediction.extend(list(current_pred_folder.glob('*.tif')))
                
                if not files_prediction:
                    print(f"WARNING: No images found in {current_pred_folder.name}. Skipping.")
                    continue
                
                output_path = current_pred_folder.parent / folder_name
                output_path.mkdir(parents=True, exist_ok=True)
                
                print(f"Found {len(files_prediction)} prediction files.")
                
                print('--- Looking for missing annotations ---')
                add_missing_files(files_prediction, anotators_files, selected_gt)
                print('--- Generating evaluation metrics ---')
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
                    staple_thresh,
                    eval_annotators,
                    use_staple,
                    use_union,
                    use_intersection
                )
                
                print('--- Generating plots ---')
                graph_generator(output_path, current_pred_folder, anotators_files)
                
                print(f"Completed: {current_pred_folder.name}")

            print('\n###################################### All evaluations completed ############################################')
            print(f'Evaluation results saved at {base_prediction_path.parent if direct_images else base_prediction_path} / {folder_name}')
            
    run_button.on_click(on_button_clicked)


def save_summary_plots(df, folder_path, is_single=True):
    if df.empty:
        return

    label_col = 'Statistic' if 'Statistic' in df.columns else 'id'
    
    if is_single and label_col == 'Statistic':
        plot_df = df[df[label_col] == 'mean']
    else:
        plot_df = df

    if len(df.columns) <= 7:
        m_jaccard = [c for c in df.columns if 'jaccard' in c.lower()]
        m_dice = [c for c in df.columns if 'dice' in c.lower()]
        m_hausdorff = [c for c in df.columns if 'hausdorff' in c.lower()]
        m_msd = [c for c in df.columns if 'msd' in c.lower()]
        m_biou = [c for c in df.columns if 'biou' in c.lower()]
        m_oca = [c for c in df.columns if 'oca' in c.lower()]

        metrics_list = []
        if m_jaccard: metrics_list.append(('Jaccard', m_jaccard[0], 'left'))
        if m_dice: metrics_list.append(('Dice', m_dice[0], 'left'))
        if m_biou: metrics_list.append(('BIoU', m_biou[0], 'left'))
        if m_oca: metrics_list.append(('OCA', m_oca[0], 'left'))
        if m_hausdorff: metrics_list.append(('Hausdorff', m_hausdorff[0], 'right'))
        if m_msd: metrics_list.append(('MSD', m_msd[0], 'right'))

        plt.figure(figsize=(12, 7))
        ax1 = plt.gca()
        ax2 = ax1.twinx() 

        n_models = len(plot_df)
        n_metrics = len(metrics_list)
        x = np.arange(n_metrics)
        width = 0.8 / n_models 

        colors = plt.cm.tab20(np.linspace(0, 1, n_models))
        legend_handles = []

        for i, (idx, row) in enumerate(plot_df.iterrows()):
            model_name = row[label_col]
            model_color = colors[i]
            
            for m_idx, (m_label, col_name, side) in enumerate(metrics_list):
                pos = m_idx + (i - n_models/2 + 0.5) * width
                val = row[col_name]
                
                if side == 'left':
                    bar = ax1.bar(pos, val, width, color=model_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                else:
                    bar = ax2.bar(pos, val, width, color=model_color, alpha=0.6, hatch='//', edgecolor='black', linewidth=0.5)
                
                if m_idx == 0:
                    legend_handles.append(plt.Rectangle((0,0),1,1, color=model_color, label=model_name))

        ax1.set_ylabel('Precision Scores (Jaccard / Dice / BIoU / OCA)↑', color='blue', fontsize=12, fontweight='bold')
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
            'Hausdorff': '_hausdorff',
            'BIoU': '_biou',
            'OCA': '_oca'
        }

        label_col = 'Statistic' if 'Statistic' in df.columns else 'id'
        
        if is_single:
            plot_df = df[df[label_col] == 'mean']
        else:
            plot_df = df

        num_curves = len(plot_df)
        colors = plt.cm.tab20(np.linspace(0, 1, num_curves))
        for title, suffix in metrics.items():
            relevant_cols = [c for c in df.columns if c.endswith(suffix)]
            
            if not relevant_cols:
                continue

            plt.figure(figsize=(12, 7))
            x_labels = [c.replace(suffix, '') for c in relevant_cols]
            
            for i, (_, row) in enumerate(plot_df.iterrows()):
                label = row[label_col]
                values = row[relevant_cols].values
                plt.plot(x_labels, values, marker='o', label=label,color=colors[i])

            if title in ['Dice', 'Jaccard', 'BIoU', 'OCA']:
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

def process_single_folder(folder_path: Path):
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
    input_csv_files = [f for f in csv_files if f.name not in output_filenames]
    
    if not input_csv_files:
        print(f"No valid .csv files found in {folder_path}")
        return

    for file in tqdm(input_csv_files, desc=f"Summary process ({folder_path.name})"):
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
        print(f"No valid input files were processed in {folder_path.name}. Exiting folder.")
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


def generate_statistical_summaries(folder_path: Path):
    output_filenames = [
        'mean_summary.csv', 
        'std_summary.csv', 
        'variance_summary.csv', 
        'min_max_summary.csv',
        'single_summary.csv' 
    ]
    
    root_csvs = [f for f in folder_path.glob('*.csv') if f.name not in output_filenames]
    
    if root_csvs:
        print(f"Found .csv files in the main folder. Processing: {folder_path.name}")
        process_single_folder(folder_path)
    else:
        subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
        
        if not subdirs:
            print(f"No .csv files or subdirectories found in {folder_path}")
            return
            
        print(f"No direct .csv files found in main folder. Checking subdirectories...")
        processed_any = False
        
        for subdir in subdirs:
            subdir_csvs = [f for f in subdir.glob('*.csv') if f.name not in output_filenames]
            if subdir_csvs:
                print(f"\n--- Processing subfolder: {subdir.name} ---")
                process_single_folder(subdir)
                processed_any = True
            else:
                print(f"Skipping {subdir.name} (no valid .csv files found)")
                
        if not processed_any:
            print("No valid .csv files found in any subdirectories either.")


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