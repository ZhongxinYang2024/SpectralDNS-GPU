from abc import ABC, abstractmethod




class TimeIntegrator(ABC):
    def __init__(self, dt):
        self.dt = dt

    @abstractmethod
    def step(self, state, equation, boundary_conditions, **kwargs):
        pass
    
    
    
class RK4(TimeIntegrator):
    def step(self, state, equation, boundary_conditions, **kwargs):
        dt = self.dt
    
        ## stage 1
        k1 = equation.compute_residual(state, boundary_conditions, **kwargs)
        
        # stage 2
        state2 = state + 0.5 * dt * k1
        boundary_conditions.apply(state2)
        k2 = equation.compute_residual(state2, boundary_conditions, **kwargs)
        
        # stage 3
        state3 = state + 0.5 * dt * k2
        boundary_conditions.apply(state3)
        k3 = equation.compute_residual(state3, boundary_conditions, **kwargs)
        
        # stage 4
        state4 = state + dt * k3
        boundary_conditions.apply(state4)
        k4 = equation.compute_residual(state4, boundary_conditions, **kwargs)


        # combine
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        boundary_conditions.apply(new_state)
        return new_state



class RK2(TimeIntegrator):
    def step(self, state, equation, boundary_conditions,  **kwargs):
        dt = self.dt
    
        ## stage 1
        k1 = equation.compute_residual(state, boundary_conditions, **kwargs)
        
        # stage 2
        state2 = state + 1.0 * dt * k1
        boundary_conditions.apply(state2)
        k2 = equation.compute_residual(state2, boundary_conditions, **kwargs)
        

        # combine
        new_state = state + (dt / 2.0) * (k1 + k2)
        boundary_conditions.apply(new_state)
        
        return new_state
