from shutil import copyfile
from pathlib import Path

src = Path("/nfs/home4/mhouben/facenet_pytorch/log/20190814-221208/checkpoint_epoch158.pth")
dest = Path("/nfs/home4/mhouben/facenet_pytorch/log/tmp/checkpoint_epoch158.pth")
copyfile(src, dest)