from utils.convnext_utils import CNBlockConfig

config = dict()

config['gpu'] = '3'
config['name'] = 'pyr_skirt'
config['notes'] = ''
config['tags'] = ['pyr_skirt']

config['batch_size'] = 4
config['is_skirt'] = True
config['is_cape'] = False
config['device'] = 'cuda'

config['pyramid'] = dict()
config['pyramid']['resolutions'] = [64, 128, 256, 512]

config['data_dir'] = '/data/cloth3d_processed'
config['out_dir'] = '/output'
config['frames'] = 'frames_test_skirt.txt'
config['dataset_in_memory'] = False

config['smpl_res'] = 14475
if config['is_cape']:
  config['smpl_res'] = 3638

config['bottleneck_scale'] = 2

config['signature']: set = { 'id',
                            'frame_nr',
                            'pose_shape',
                            'uv_body_posed',
                            'mesh_body_posed',
                            'mesh_unposed',
                            'uv_unposed',
                            'uv_normals',
                            'uv_static',
                            'mesh_mask',
                            'uv_mask',
                        }

config['pyramid']['is_label'] = False

for res in config['pyramid']['resolutions']:
    if res != 512:
        res_modifier = '_' + str(res)
        config['signature'].add('uv_body_posed' + res_modifier)
        config['signature'].add('uv_unposed' + res_modifier)
        config['signature'].add('uv_normals' + res_modifier)
        config['signature'].add('uv_static' + res_modifier)
        config['signature'].add('uv_mask' + res_modifier)


config['pyramid']['levels'] = []
for res in config['pyramid']['resolutions']:
  if res == 512:
    level = dict()
    level['resolution'] = 512
    level['kernel_size'] = 7
    level['bottleneck_size'] = 64 * config['bottleneck_scale']
    level['input_features'] = 12

    block_setting = [
        CNBlockConfig(16, 32, 3),
        CNBlockConfig(32, 64, 3),
        CNBlockConfig(64, 128, 3),
        CNBlockConfig(128, level['bottleneck_size'], 9),
        CNBlockConfig(level['bottleneck_size'], None, 3),
    ]

    block_setting_reverse = [
        CNBlockConfig(level['bottleneck_size'] * 2, 128, 3),
        CNBlockConfig(128, 64, 9),
        CNBlockConfig(64, 32, 3),
        CNBlockConfig(32, 16, 3),
        CNBlockConfig(16, 3, 3),
    ]

    level['encoder_block_settings'] = block_setting
    level['decoder_block_settings'] = block_setting_reverse
    config['pyramid']['levels'].append(level)
  elif res == 256:
    level = dict()
    level['resolution'] = 256
    level['kernel_size'] = 5
    level['bottleneck_size'] = 64 * config['bottleneck_scale']
    level['input_features'] = 12

    block_setting = [
        CNBlockConfig(64, 128, 3),
        CNBlockConfig(128, 128, 3),
        CNBlockConfig(128, level['bottleneck_size'], 9),
        CNBlockConfig(level['bottleneck_size'], None, 3),
    ]

    block_setting_reverse = [
        CNBlockConfig(level['bottleneck_size'] * 2, 128, 3),
        CNBlockConfig(128, 128, 9),
        CNBlockConfig(128, 64, 3),
        CNBlockConfig(64, 3, 3),
    ]

    level['encoder_block_settings'] = block_setting
    level['decoder_block_settings'] = block_setting_reverse
    config['pyramid']['levels'].append(level)
  elif res == 128:
    level = dict()
    level['resolution'] = 128
    level['kernel_size'] = 5
    level['bottleneck_size'] = 64 * config['bottleneck_scale']
    level['input_features'] = 12

    block_setting = [
        CNBlockConfig(64, 128, 9),
        CNBlockConfig(128, level['bottleneck_size'], 9),
        CNBlockConfig(level['bottleneck_size'], None, 9),
    ]

    block_setting_reverse = [
        CNBlockConfig(level['bottleneck_size'] * 2, 128, 9),
        CNBlockConfig(128, 64, 9),
        CNBlockConfig(64, 3, 9),
    ]

    level['encoder_block_settings'] = block_setting
    level['decoder_block_settings'] = block_setting_reverse
    config['pyramid']['levels'].append(level)
  elif res == 64:
    level = dict()
    level['resolution'] = 64
    level['kernel_size'] = 3
    level['bottleneck_size'] = 64 * config['bottleneck_scale']
    level['input_features'] = 12

    block_setting = [
        CNBlockConfig(64, level['bottleneck_size'], 9),
        CNBlockConfig(level['bottleneck_size'], None, 9),
    ]

    block_setting_reverse = [
        CNBlockConfig(level['bottleneck_size'] * 2, 64, 9),
        CNBlockConfig(64, 3, 9),
    ]

    level['encoder_block_settings'] = block_setting
    level['decoder_block_settings'] = block_setting_reverse
    config['pyramid']['levels'].append(level)
  else:
    raise NotImplementedError


####################### cond_ae_norm ############################
config['cond_ae_norm'] = dict()
config['cond_ae_norm']['device'] = config['device']
config['cond_ae_norm']['kernel_size'] = 7
config['cond_ae_norm']['bottleneck_size'] = 40 * config['bottleneck_scale']

block_setting = [
    CNBlockConfig(16, 32, 3),
    CNBlockConfig(32, 64, 3),
    CNBlockConfig(64, 128, 9),
    CNBlockConfig(128, config['cond_ae_norm']['bottleneck_size'], 9),
    CNBlockConfig(config['cond_ae_norm']['bottleneck_size'], None, 3),
]

block_setting_reverse = [
    CNBlockConfig(config['cond_ae_norm']['bottleneck_size'], 128, 3),
    CNBlockConfig(128, 64, 9),
    CNBlockConfig(64, 32, 9),
    CNBlockConfig(32, 16, 3),
    CNBlockConfig(16, 3, 3),
]

config['cond_ae_norm']['encoder_block_settings'] = block_setting
config['cond_ae_norm']['decoder_block_settings'] = block_setting_reverse
config['cond_ae_norm']['activation'] = 'tanh'


####################### cond_ae_body ############################
config['cond_ae_body'] = dict()
config['cond_ae_body']['device'] = config['device']
config['cond_ae_body']['kernel_size'] = 7
config['cond_ae_body']['bottleneck_size'] = 16 * config['bottleneck_scale']

block_setting = [
    CNBlockConfig(16, 32, 3),
    CNBlockConfig(32, 64, 3),
    CNBlockConfig(64, 128, 9),
    CNBlockConfig(128, config['cond_ae_body']['bottleneck_size'], 9),
    CNBlockConfig(config['cond_ae_body']['bottleneck_size'], None, 3),
]

block_setting_reverse = [
    CNBlockConfig(config['cond_ae_body']['bottleneck_size'], 128, 3),
    CNBlockConfig(128, 64, 9),
    CNBlockConfig(64, 32, 9),
    CNBlockConfig(32, 16, 3),
    CNBlockConfig(16, 3, 3),
]

config['cond_ae_body']['encoder_block_settings'] = block_setting
config['cond_ae_body']['decoder_block_settings'] = block_setting_reverse
config['cond_ae_body']['activation'] = 'tanh'

####################### cond_ae_static ############################
config['cond_ae_static'] = dict()
config['cond_ae_static']['device'] = config['device']
config['cond_ae_static']['kernel_size'] = 7
config['cond_ae_static']['bottleneck_size'] = 8 * config['bottleneck_scale']

block_setting = [
    CNBlockConfig(8, 16, 3),
    CNBlockConfig(16, 32, 3),
    CNBlockConfig(32, 64, 3),
    CNBlockConfig(64, config['cond_ae_static']['bottleneck_size'], 9),
    CNBlockConfig(config['cond_ae_static']['bottleneck_size'], None, 3),
]

block_setting_reverse = [
    CNBlockConfig(config['cond_ae_static']['bottleneck_size'], 64, 3),
    CNBlockConfig(64, 32, 9),
    CNBlockConfig(32, 16, 3),
    CNBlockConfig(16, 8, 3),
    CNBlockConfig(8, 3, 3),
]

config['cond_ae_static']['encoder_block_settings'] = block_setting
config['cond_ae_static']['decoder_block_settings'] = block_setting_reverse
config['cond_ae_static']['activation'] = 'tanh'