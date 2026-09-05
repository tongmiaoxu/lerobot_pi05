# Vendored from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Upstream commit: 2a7afba2895d52556dd5dfe07e8555ef657ced6f
#
# Trimmed to the pix2pix path only (dropped cyclegan/colorization/unaligned/single
# dataset+model files). All copied files are otherwise unmodified, except:
# - data/sim2real_dataset.py is new: reads already-split train_A/train_B (or
#   test_A/test_B) folders directly instead of the upstream `aligned` dataset's
#   concatenated A|B images. Use `--dataset_mode sim2real`.
#
# This vendored tree uses upstream's own absolute imports (`import data`,
# `from models import create_model`, ...), so it must be run as a script
# (`python sim2real/pix2pix/train.py ...`), not imported as a package.
