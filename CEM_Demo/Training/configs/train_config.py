from .base_config import BaseConfigures
import sys


class TrainConfigures(BaseConfigures):
    def initialize(self, parser):
        BaseConfigures.initialize(self, parser)
        # for displays
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # Logger  # TODO: load configurations for logger
        parser.add_argument('--display_freq', type=int, default=1000000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--test_freq', type=int, default=30, help='frequency of showing testing psnr on console')
        parser.add_argument('--save_latest_freq', type=int, default=15000, help='frequency of saving the latest results')
        parser.add_argument('--num_vis_images', type=int, default=8, help='print images in vis')

        # build loss
        parser.add_argument('--sr_loss_mode', type=str, default='l1', help='The SR content loss type, [l1, l2]')
        # train phase
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=10, help='input batch size')
        parser.add_argument('--generator_lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=5000, help='number of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=10000, help='number of iter to linearly decay learning rate to zero')
        parser.add_argument('--decay_round', type=int, default=5, help='number of iter to linearly decay learning rate to zero')

        # train optimizing
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme, check paper: <GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium>')

        # the default values for beta1 and beta2 differ by TTUR option
        # opt, _ = parser.parse_known_args()
        # if opt.no_TTUR:
        #     parser.set_defaults(beta1=0.5, beta2=0.999)

        self.isTrain = True
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # train dataset
        parser.add_argument('--lr_size', type=int, default=64, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--v_flip', type=float, default=0.5, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--h_flip', type=int, default=0.5, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')

        # for setting inputs
        parser.add_argument('--hr_dataroot', type=str, default='/mnt/lustre/gujinjin1/IRSaliency/Training/utils/HR/DIV2K_train_HR/x4')
        parser.add_argument('--lr_dataroot', type=str, default='/mnt/lustre/gujinjin1/IRSaliency/Training/utils/LR/LRBI/DIV2K_train_HR/x4')
        # parser.add_argument('--hr_dataroot', type=str, default='/mnt/lustre/gujinjin1/SRData/DIV2K_train_HR')
        # parser.add_argument('--lr_dataroot', type=str, default='/mnt/lustre/gujinjin1/SRData/DIV2K_train_LR4')
        parser.add_argument('--test_set', type=str, default='/mnt/lustre/gujinjin1/SRData/BSD100')
        parser.add_argument('--dataset_mode', type=str, default='lmdb')

        return parser



