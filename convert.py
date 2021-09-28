from utils import pth2pdparams
from model import FarSeg


pth_path = "farseg50.pth"
model = FarSeg(num_classes=16)
pth2pdparams(model, pth_path)