import torch


def compute_grad(f, x, *args, **kwargs):
    """
    Compute the gradient of f(x, *args, **kwargs) with respect to x.
    
    Args:
        f: Callable function.
        x: Input tensor with respect to which the gradient is computed.
        args, kwargs: Additional function arguments.
        
    Returns:
        grad_x: Gradient with the same shape as x.
    """
    # Ensure that x tracks gradients.
    x_input = x.clone().detach().requires_grad_(True)
    
    # Evaluate the function.
    f_value = f(x_input, *args, **kwargs)
    
    # Check whether f_value is a scalar.
    if f_value.ndim == 0:
        # Scalar output: backpropagate directly.
        f_value.backward()
        grad_x = x_input.grad
    else:
        # Non-scalar output: accumulate gradients for all output components.
        # Create a tensor of ones with the same shape as f_value.
        grad_outputs = torch.ones_like(f_value, device=f_value.device)
        
        # Compute the gradient.
        grad_x = torch.autograd.grad(
            outputs=f_value,
            inputs=x_input,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
    
    return grad_x


def test_compute_grad_batch():
    import torch
    import torch.nn as nn

    # Test configuration.
    batch_size = 5
    input_dim = 3
    output_dim = 1
    
    # Create the model and inputs.
    model = nn.Linear(input_dim, output_dim)
    x = torch.randn(batch_size, input_dim)  # Batched input.
    
    # Test 1: scalar-valued function (one loss for the full batch).
    def scalar_func(x):
        outputs = model(x)
        return outputs.pow(2).mean()  # Return a scalar loss.
    
    grad_scalar = compute_grad(scalar_func, x)
    
    # Validate the scalar-output gradient.
    assert grad_scalar.shape == x.shape, "Incorrect scalar-output gradient shape"
    
    # Validate against PyTorch autograd.
    x.requires_grad_(True)
    loss = scalar_func(x)
    loss.backward()
    torch_grad_scalar = x.grad
    assert torch.allclose(grad_scalar, torch_grad_scalar, atol=1e-6), "Incorrect scalar-output gradient values"
    
    print("Test 1 passed: scalar-output gradient is correct")
    
    # Test 2: vector-valued function (one output per sample).
    def vector_func(x):
        return model(x)  # Return a tensor shaped [batch_size, output_dim].
    
    grad_vector = compute_grad(vector_func, x)
    
    # Validate the vector-output gradient.
    assert grad_vector.shape == x.shape, "Incorrect vector-output gradient shape"
    
    # Validate against PyTorch autograd.
    x.requires_grad_(True)
    outputs = vector_func(x)
    grad_outputs = torch.ones_like(outputs)
    torch_grad_vector = torch.autograd.grad(
        outputs, x, grad_outputs=grad_outputs, create_graph=False
    )[0]
    assert torch.allclose(grad_vector, torch_grad_vector, atol=1e-6), "Incorrect vector-output gradient values"
    
    print("Test 2 passed: vector-output gradient is correct")
    
if __name__ == '__main__':
    test = 3;
    
    match test:
        case 1:
            # Example 1: scalar-valued function.
            def f_scalar(x):
                # Return a scalar by summing over the batch.
                return torch.sum(x**2)

            # Example 2: batched-output function.
            def f_vector(x):
                # Return one result per sample with shape [batch_size].
                return x[:, 0]**2 + 3*x[:, 1]

            # Batched input with batch_size=3.
            x_batch = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            # Compute the scalar-function gradient.
            grad_scalar = compute_grad(f_scalar, x_batch)
            print("Scalar-function gradient:")
            print(grad_scalar)  # Expected batched form: [2x, 2y].

            # Compute the vector-function gradient.
            grad_vector = compute_grad(f_vector, x_batch)
            print("\nVector-function gradient:")
            print(grad_vector)  # Expected batched form: [[2x, 3], ...].
        case 2:
            def Compute_Saction(x, x_00, x_neg1, T):
                """
                Compute trajectory velocities and the action.
                
                Args:
                x: Intermediate positions shaped (batch_size, num_steps, spatial_dim).
                x_00: Initial positions shaped (batch_size, spatial_dim).
                x_neg1: Final positions shaped (batch_size, spatial_dim).
                T: Total duration.
                
                Returns:
                Saction: Action tensor shaped (batch_size,).
                """
                # Ensure that all inputs are tensors.
                assert all(isinstance(t, torch.Tensor) for t in [x, x_00, x_neg1]), "All inputs must be torch.Tensor objects"
                
                # Add the initial and final positions to the trajectory.
                full_x = torch.cat([
                    x_00.unsqueeze(1),  # Add a time dimension and prepend.
                    x,
                    x_neg1.unsqueeze(1)  # Add a time dimension and append.
                ], dim=1)
                
                # Count all time steps, including the endpoints.
                total_steps = full_x.shape[1]
                
                # Divide the total duration by the number of intervals.
                dt = T / (total_steps - 1)  # Time-step size.
                
                # Compute velocity directly from differences of full_x.
                
                # Forward difference: x_{t+1} - x_t.
                forward_diff = full_x[:, 1:] - full_x[:, :-1]
                
                # Centered difference: x_{t+1} - x_{t-1}.
                center_diff = full_x[:, 2:] - full_x[:, :-2]
                
                # Initialize the velocity tensor.
                v = torch.empty_like(full_x)
                
                # Use one-sided differences at the endpoints.
                v[:, 0] = forward_diff[:, 0] / dt  # Initial point (forward difference).
                v[:, -1] = forward_diff[:, -1] / dt  # Final point (backward difference).
                
                # Use centered differences at interior points.
                if total_steps > 2:
                    v[:, 1:-1] = center_diff / (2 * dt)
                
                # Compute kinetic energy for unit mass.
                Kenergy = 0.5 * v ** 2
                
                # Compute potential energy.
                Penergy = 0.5 * full_x ** 2
                
                # Integrate the Lagrangian over time to obtain the action.
                Saction = torch.sum((Kenergy - Penergy) * dt, dim=1)
                
                return Saction


            def CondScore(x,T):
                Batch_size = x.shape[0]

                x_00 = torch.ones((Batch_size,),dtype=torch.float32, device=x.device)*0.0
                x_neg1 = torch.ones((Batch_size, ), dtype=torch.float32, device=x.device)*0.0
                Cgrad = compute_grad(Compute_Saction, x, x_00, x_neg1, T)
                return Cgrad

            x = torch.ones(16,2)
            grad_conditional = CondScore(x,1)
            print(grad_conditional[0])
        case 3:
            test_compute_grad_batch()
