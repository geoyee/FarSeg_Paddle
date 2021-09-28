import torch
import pickle
from collections import OrderedDict


def pth2pdparams(model, pth_path):
    pd_pw = model.state_dict()
    pt_pw = torch.load(pth_path)["model"]  # pth参数
    pt_pw_tmp = pt_pw.copy()
    pd_new_dict = OrderedDict()
    # 删除没有的
    for k in pt_pw_tmp.keys():
        sq = k.split(".")
        if sq[-1] in ["buffer_step", "num_batches_tracked"]:
            pt_pw.pop(k)
    print(len(pt_pw), len(pd_pw))
    for kt, kp in zip(pt_pw.keys(), pd_pw.keys()):
        # print(kt, kp)
        # if "fc" in kp.split("."):
        #     pd_new_dict[kp] = pt_pw[kp].detach().numpy().T
        # else:
        pd_new_dict[kp] = pt_pw[kt].detach().numpy()
    pdparams_path = pth_path.split(".")[0] + '.pdparams'
    with open(pdparams_path, 'wb') as f:
        pickle.dump(pd_new_dict, f)
    print('\nConvert finished!')
    return pdparams_path