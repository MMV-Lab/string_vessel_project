import napari
import numpy as np
from pathlib import Path
import argparse 

def parse_args():

    parser = argparse.ArgumentParser(
        description="Visualizer",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--path',
        type=str,
        required=True, 
        help='Path to the folder with the .npy files.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    
    full_path = Path(args.path)
    
    
    file_id = full_path.name
    
 
    base_path = full_path

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