import torch
import torch.nn.functional as F
from torchvision import transforms
from .base_model import BaseModel
from . import networks


def _forward_dinov3(net_dinov3, pixel_values):
    try:
        return net_dinov3(pixel_values=pixel_values, interpolate_pos_encoding=True)
    except TypeError:
        return net_dinov3(pixel_values=pixel_values)


def _dinov3_patch_features(net_dinov3, images, normalize, device):
    # images are in [-1, 1] (pix2pix's Normalize(0.5, 0.5, 0.5)); DINOv3 expects ImageNet-normalized [0, 1].
    images = normalize(images * 0.5 + 0.5)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = _forward_dinov3(net_dinov3, images)
    features = outputs.last_hidden_state
    num_prefix_tokens = 1 + int(getattr(net_dinov3.config, "num_register_tokens", 0))
    return features[:, num_prefix_tokens:, :].float()


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")
            parser.add_argument(
                "--lambda_dinov3_pixel",
                type=float,
                default=0.0,
                help="Weight for pixel-aligned L2 between DINOv3 patch features of real_A (sim input) and "
                "fake_B (generated image). 0 (default) disables DINOv3 entirely, matching original pix2pix.",
            )
            parser.add_argument(
                "--dinov3_model_name",
                type=str,
                default="facebook/dinov3-vits16-pretrain-lvd1689m",
                help="Hugging Face model id or local path for the frozen DINOv3 feature extractor.",
            )
            parser.add_argument(
                "--dinov3_trust_remote_code",
                action="store_true",
                help="Pass trust_remote_code=True when loading the DINOv3 model from Hugging Face.",
            )

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        self.device = opt.device
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # move to the device for custom loss
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Optional DINOv3 pixel-aligned structural loss (off by default; original pix2pix is unaffected).
            self.net_dinov3 = None
            if getattr(opt, "lambda_dinov3_pixel", 0.0) > 0:
                from transformers import AutoModel

                self.loss_names.append("G_DINOv3")
                self.net_dinov3 = AutoModel.from_pretrained(
                    opt.dinov3_model_name,
                    trust_remote_code=opt.dinov3_trust_remote_code,
                ).to(self.device)
                self.net_dinov3.requires_grad_(False)
                self.net_dinov3.eval()
                self.dinov3_renorm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.net_dinov3 is not None:
            with torch.no_grad():
                dinov3_src_features = _dinov3_patch_features(self.net_dinov3, self.real_A.detach(), self.dinov3_renorm, self.device)
            dinov3_pred_features = _dinov3_patch_features(self.net_dinov3, self.fake_B, self.dinov3_renorm, self.device)
            self.loss_G_DINOv3 = F.mse_loss(dinov3_pred_features, dinov3_src_features, reduction="mean") * self.opt.lambda_dinov3_pixel
            self.loss_G = self.loss_G + self.loss_G_DINOv3
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights
