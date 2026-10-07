import os
import numpy as np

def make_dir(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the directory.
        print(f"make {folder_path}")
        

def save_to_tecplot(filename, ll, variables, var_names):
    """
    Use PyTecplot to save structed grid data
    
    Arg:
        filename: 
        x, y, z: grid coordinate (nx, ny, nz)
        variables -> list: shape (nx, ny, nz)
        var_names -> list: 
    """
    nn = ll.shape[0]
    
    with open(filename, 'w') as f:
        f.write('TITLE = "FlowField"\n')

        ## write variables name
        f.write('VARIABLES = "X", "Y", "Z"')
        for name in var_names:
            f.write(f', "{name}"')
        f.write('\n')
        
        # write zone
        f.write(f'ZONE')
        f.write(f'  I={nn} J={nn} K={nn}\n')
        
        for k in range(nn):
            for j in range(nn):
                for i in range(nn):
                    f.write(f"{ll[i]:.6e}    {ll[j]:.6e}    {ll[k]:.6e}")
                    
                    for iv, var in enumerate(var_names):
                        f.write(f"    {variables[iv][i,j,k]:.6e}")
                    f.write('\n')
    

def save_to_tecplot_fast(filename, ll, variables, var_names):            
    nn = ll.shape[0]
    
    x, y, z = np.meshgrid(ll, ll, ll, indexing='ij')
    
    with open(filename, 'w') as f:
        f.write('TITLE = "FlowField"\n')
        
        # Write variables name
        f.write('VARIABLES = "X", "Y", "Z"')
        for name in var_names:
            f.write(f', "{name}"')
        f.write('\n')
        
        # Write zone
        f.write(f'ZONE I={nn} J={nn} K={nn}\n')
        
        # Flatten all arrays and combine
        all_data = np.column_stack([
            np.ravel(x, order='F'), 
            np.ravel(y, order='F'), 
            np.ravel(z, order='F'), 
            *[np.ravel(var, order='F') for var in variables]
        ])
        
        # Format all numbers at once
        np.savetxt(f, all_data, fmt='%.6e', delimiter='    ')
