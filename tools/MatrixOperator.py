import torch

def Cal_Stat(U, priodic_dim):
    U_avg = U
    Reduced_dim = 0
    
    for i in range(len(priodic_dim)):
        # torch.mean corresponds to np.average(..., axis=).
        # keepdim=False removes the reduced dimension, matching NumPy behavior.
        U_avg = torch.mean(U_avg, dim=priodic_dim[i] - Reduced_dim)
        Reduced_dim += 1
    
    return U_avg

def TransferMatrix(Matrix, steps):
    if steps >= 0:
        New_Matrix = torch.cat((Matrix[steps:], Matrix[:steps]))
    else:
        steps = abs(steps)
        New_Matrix = torch.cat((Matrix[-steps:], Matrix[:-steps]))
    
    return New_Matrix

def TransferMatrix_alongAxis(Matrix, axis, steps):
    return torch.roll(Matrix, shifts=-steps, dims=axis)


if __name__ == "__main__":
    # Test Cal_Stat.
    U = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]], dtype=torch.float32)
    print("Original shape:", U.shape)
    U_avg = Cal_Stat(U, [0, 1])  # Average over dimensions 0 and 1.
    print("Shape after averaging:", U_avg.shape)
    print("Average:", U_avg)

    # Test TransferMatrix.
    arr = torch.tensor([1, 2, 3, 4, 5])
    print("Original array:", arr)
    print("Shift forward by 2:", TransferMatrix(arr, 2))
    print("Shift backward by 1:", TransferMatrix(arr, -1))

    # Test TransferMatrix_alongAxis.
    matrix = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    print("\nOriginal matrix:")
    print(matrix)
    print("\nShift by 1 along axis 0:")
    print(TransferMatrix_alongAxis(matrix, axis=0, steps=1))
    print("\nShift by -1 along axis 1:")
    print(TransferMatrix_alongAxis(matrix, axis=1, steps=-1))
    
