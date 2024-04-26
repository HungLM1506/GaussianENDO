from poses.pose_utils import gen_poses
import sys

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--match_type', type=str,
                    default='exhaustive_matcher',
                    help='type of matcher used. Valid options: \
                            exhaustive_matcher sequential_matcher. \
                            Other matchers not supported at this time')

parser.add_argument('--use_gpu', type=str,
                    default='0',
                    choices=['0', '1'],
                    help='whether to use GPU acceleration for \
                    COLMAP feature extraction and matching. Valid options: \
                    enable 1, disable 0. (Default 0)')

parser.add_argument('--num_threads', type=str,
                    default='30',
                    help='Number of CPU threads to use for \
                    COLMAP feature extraction and matching.')

parser.add_argument('--camera_model', type=str,
                    default='SIMPLE_RADIAL',
                    choices=['SIMPLE_RADIAL', 'SIMPLE_PINHOLE',
                             'PINHOLE', 'RADIAL', 'OPENCV', 'FULL_OPENCV'],
                    help='Camera model for feature extraction (default SIMPLE_RADIAL)')

parser.add_argument('scenedir', type=str,
                    help='input scene directory')

args = parser.parse_args()

if args.match_type != 'exhaustive_matcher' and \
   args.match_type != 'sequential_matcher':
    print('ERROR: matcher type ' + args.match_type + ' is not valid.\
           Aborting')
    sys.exit()

if __name__ == '__main__':
    gen_poses(args.scenedir, args.match_type, args.use_gpu,
              args.num_threads, args.camera_model)
