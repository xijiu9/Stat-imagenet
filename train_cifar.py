import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected in train_imagenet.')

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--twolayers_gradweight', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--twolayers_gradinputt', type=str2bool, default=False, help='use two 4 bit to simulate a 8 bit')
parser.add_argument('--lsqforward', type=str2bool, default=False, help='apply LSQ')

parser.add_argument('--training-bit', type=str, default='', help='weight number of bits',
                    choices=['exact', 'qat', 'all8bit', 'only_weight', 'all4bit', 'forward8', 'forward4'])
parser.add_argument('--training-strategy', default='scratch', type=str, metavar='strategy',
                    choices=['scratch', 'checkpoint', 'checkpoint_from_zero', 'checkpoint_full_precision'])
parser.add_argument('--checkpoint_epoch_full_precision', type=int, default=0, help='full precision')
parser.add_argument('--clip-grad', type=str2bool, default=False, help='clip gradient to 0.01(CIFAR)')

parser.add_argument('--lr', type=float, default=0.1, help='clip gradient to 0.01(CIFAR)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--warmup', default=20, type=int, metavar='E', help='number of warmup epochs')
args = parser.parse_args()

arg = " -c quantize --qa=True --qw=True --qg=True"

if args.training_bit == 'all8bit':
    bbits, bwbits, awbits = 8, 8, 8
elif args.training_bit == 'exact':
    arg = ''
    bbits, bwbits, awbits = 0, 0, 0
elif args.training_bit == 'qat':
    bbits, bwbits, awbits = 0, 0, 8
    arg = "-c quantize --qa=True --qw=True --qg=False"
elif args.training_bit == 'only_weight':
    bbits, bwbits, awbits = 8, 4, 4
elif args.training_bit == 'all4bit':
    bbits, bwbits, awbits = 4, 4, 4
elif args.training_bit == 'forward8':
    bbits, bwbits, awbits = 4, 4, 8
elif args.training_bit == 'forward4':
    bbits, bwbits, awbits = 8, 8, 4
else:
    bbits, bwbits, awbits = 0, 0, 0
    print("!"*1000)

if args.twolayers_gradweight:
    assert bwbits == 4
if args.twolayers_gradinputt:
    assert bbits == 4

if args.twolayers_gradweight and args.twolayers_gradinputt:
    method = 'twolayer'
elif args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = 'twolayer_weightonly'
elif not args.twolayers_gradweight and not args.twolayers_gradinputt:
    method = args.training_bit

if args.training_strategy == 'checkpoint' or args.training_strategy == 'checkpoint_from_zero':
    model = ''
elif args.training_strategy == 'checkpoint_full_precision':
    model = ''
else:
    model = 'pass'

os.system("python ./main.py --arch preact_resnet56 --gather-checkpoints \
            --lr {} --resume {} --dataset cifar10 --momentum 0.9 --weight-decay {} --epoch 200\
            --warmup {} {}  ~/data/cifar10 --workspace ./results/cifar/{}/models \
            --amp --static-loss-scale 128 --print-freq 100 \
            --bbits {} --bwbits {} --abits {} --wbits {}"
            .format(args.lr, model, args.weight_decay,
                    args.warmup, arg, args.training_bit,
                    bbits, bwbits, awbits, awbits))