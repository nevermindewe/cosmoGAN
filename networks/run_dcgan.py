import os

# These will be the default parameters for the DCGAN
config_kwargs = dict(
    datafile = 'data/cosmogan_maps_256_8k_1.npy',
    output_size = 256,
    epoch = 50,
    flip_labels = 0.01,
    batch_size = 128,
    z_dim = 64,
    nd_layers = 4,
    ng_layers = 4,
    gf_dim = 64,
    df_dim = 64,
    save_every_step = 'False',
    data_format = 'NCHW',
    transpose_matmul_b = False,
    verbose = 'True',
)

config_kwargs['experiment'] = ('cosmo_myExp_batchSize%i_flipLabel%0.3f_nd%i_ng%i_gfdim%i_dfdim%i_zdim%i' %
                               (config_kwargs['batch_size'],
                                config_kwargs['flip_labels'],
                                config_kwargs['nd_layers'],
                                config_kwargs['ng_layers'],
                                config_kwargs['gf_dim'],
                                config_kwargs['df_dim'],
                                config_kwargs['z_dim']))


if __name__ == "__main__":
    import models.main
    import sys

    # The default is to log stdout to an output file.
    # Only if the user gives the "--stdout" command line option
    #   should we display stdout to the real stdout.
    _old_stdout = sys.stdout
    _new_stdout = sys.stdout

    if "--stdout" in sys.argv:
        sys.argv.remove("--stdout")

    else:        
        if not os.path.isdir('output'):
            os.mkdir('output')
            
        _new_stdout = open(os.path.join("output", config_kwargs['experiment']), 'wb')
        

    try:
        sys.stdout = _new_stdout
        models.main.main(**config_kwargs)
    finally:
        sys.stdout = _old_stdout
