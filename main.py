from HIT3D import HITsolver3D_effective
import torch


if __name__ == '__main__':
    from configs import config_N128_ReT100 as configA
    config = configA.get_config()
    torch.manual_seed(config.get('seed', 0))
    rule = 'sim'  # Change to 'post' to process saved snapshots.

    HITsolver3D_effective(
        config,
        rule,
        U_prev=None,
        last_time=0,
        num_stat=300,
        batch_size_stat=10,
    )
    

    
