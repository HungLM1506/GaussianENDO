import os
import subprocess

# As shell script:

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense


def run_colmap(basedir, match_type, use_gpu, num_threads, camera_model):

    logfile_name = os.path.join(basedir, 'colmap_output.txt')

    logfile = open(logfile_name, 'w')
    feature_extractor_args = [
        'colmap', 'feature_extractor',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', camera_model,
        '--SiftExtraction.use_gpu', use_gpu
    ]
    feat_output = subprocess.Popen(feature_extractor_args,
                                   stdout=subprocess.PIPE)
    for line in iter(feat_output.stdout.readline, b''):
        logfile.write(line.decode('utf-8'))
    print('Features extracted')
    logfile.close()

    logfile = open(logfile_name, 'a')
    exhaustive_matcher_args = [
        'colmap', match_type,
        '--database_path', os.path.join(basedir, 'database.db'),
        '--SiftMatching.num_threads', num_threads,
        '--SiftMatching.use_gpu', use_gpu
    ]
    match_output = subprocess.Popen(exhaustive_matcher_args,
                                    stdout=subprocess.PIPE)
    for line in iter(match_output.stdout.readline, b''):
        logfile.write(line.decode('utf-8'))
    print('Features matched')
    logfile.close()

    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    logfile = open(logfile_name, 'a')
    mapper_args = [
        'colmap', 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', os.path.join(basedir, 'sparse'),
        # --export_path changed to --output_path in colmap 3.6
        # '--Mapper.init_min_tri_angle', '4',
        # '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '0',
        '--Mapper.num_threads', num_threads
    ]
    map_output = subprocess.Popen(mapper_args,
                                  stdout=subprocess.PIPE)
    for line in iter(map_output.stdout.readline, b''):
        logfile.write(line.decode('utf-8'))
    print('Sparse map created')
    logfile.close()

    print('Finished running COLMAP, see {} for logs'.format(logfile_name))
