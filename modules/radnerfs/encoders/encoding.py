from modules.radnerfs import SHEncoder
from modules.radnerfs import FreqEncoder
from modules.radnerfs import GridEncoder

def get_encoder(encoding, input_dim=3,
                multires=6, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                interpolation='linear',
                **kwargs):
    encoder = None
    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim
    
    elif encoding == 'frequency':
        #from modules.radnerfs.encoders.freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'spherical_harmonics':
        #print("start imported--Tom Chen")
        # from modules.radnerfs.encoders.shencoder import SHEncoder
        # print("imported--Tom Chen")
        encoder = SHEncoder(input_dim=input_dim, degree=degree)
        # print("encoder--Tom Chen")
    elif encoding == 'hashgrid':
        # from modules.radnerfs.encoders.gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners, interpolation=interpolation, **kwargs)
    elif encoding == 'tiledgrid':
        from modules.radnerfs import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners, interpolation=interpolation, **kwargs)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, spherical_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim