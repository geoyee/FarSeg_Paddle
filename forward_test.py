import paddle
from model import FarSeg
import torch
from pytorch.module import farseg
from pytorch.configs.isaid import farseg50
from simplecv.util import registry
from simplecv.util import checkpoint


cfg = farseg50.config["model"]
pth_path = "farseg50.pth"
tc_model = registry.MODEL[cfg["type"]](cfg['params'])
tc_params = checkpoint.load_model_state_dict_from_ckpt(pth_path)
tc_model.load_state_dict(tc_params)
tc_model.eval()

pdparams_path = "farseg50.pdparams"
pd_model = FarSeg(num_classes=16)
pd_params = paddle.load(pdparams_path)
pd_model.set_state_dict(pd_params)
pd_model.eval()

tc_data = torch.ones(1, 3, 256, 256)
pd_data = paddle.ones([1, 3, 256, 256])
print(tc_data.shape, pd_data.shape)
tc_pread = tc_model(tc_data)
pd_pread = pd_model(pd_data)
print("torch :", tc_pread.detach().numpy()[0, 0, 0, :6])
print("paddle:", pd_pread[0].numpy()[0, 0, 0, :6])