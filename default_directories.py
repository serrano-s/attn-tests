base_output_dir = 'attn-test-output/'
base_serialized_models_dir = 'models/'
base_data_dir = 'data/'
dir_with_config_files = 'configs/'
images_dir = 'imgs/'
tex_files_dir = 'generated_tex_files/'
vocabs_dir = 'vocabs/'

all_dirs = list(globals().keys())
for dirname in all_dirs:
    if not dirname.startswith('__') and not globals()[dirname].endswith('/'):
        globals()[dirname] = globals()[dirname] + '/'
