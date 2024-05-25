import torch


def get_neigbor_tensors(X: torch.Tensor, n: int = 4, entrophy=False):
    """
    Args:
        X: tensor of shape (B, C, H, W)
        n: number of neighbors to get 4 or 8
    Returns:
        list of tensors of shape (B, C, H, W)
    """
    assert n in [4, 8]
    if entrophy:
        X = X.unsqueeze(1)
        X = torch.nn.functional.pad(X, (1, 1, 1, 1, 0, 0, 0, 0)).squeeze(1)
    else:
        X = X.unsqueeze(1)
        X = torch.nn.functional.pad(X, (1, 1, 1, 1, 0, 0, 0, 0))

    neigbors = []
    if n == 8:
        for i, j in [(None, -2), (1, -1), (2, None)]:
            for k, l in [(None, -2), (1, -1), (2, None)]:
                if i == k and i == 1:
                    continue
                if entrophy:
                    neigbors.append(X[:, i:j, k:l])
                else:
                    neigbors.append(X[:, :, i:j, k:l])
    elif n == 4:
        if entrophy:
            neigbors.append(X[:, :-2, 1:-1])
            neigbors.append(X[:, 2:, 1:-1])
            neigbors.append(X[:, 1:-1, :-2])
            neigbors.append(X[:, 1:-1, 2:])
        else:
            neigbors.append(X[:, :, :-2, 1:-1])
            neigbors.append(X[:, :, 2:, 1:-1])
            neigbors.append(X[:, :, 1:-1, :-2])
            neigbors.append(X[:, :, 1:-1, 2:])
    return neigbors


def expected_neighbourhoods(cfg):
    pascal_e_joint = [0.9980886695132567, 0.9860495622941258, 0.9555923573386919, 0.9888166500147213,
                      0.9861891841249937, 0.9894536447668415, 0.9928481727772317, 0.9899944432748093, 0.9939554498312092,
                      0.9723734804881907, 0.9814734877790432, 0.992167780556063, 0.9921082109870506, 0.9904260573555301,
                      0.9877020724598903, 0.9890704556150036, 0.9771896568239452, 0.9865771364049042, 0.9878175644560833,
                      0.9948415228771397, 0.993315147276186]
    cityscapes_e_joint = [0.9964431323326034, 0.9861639854327645, 0.961200350864672, 0.831981383888851,
                          0.9733805255682632, 0.9763628214927057, 0.9914859783638212, 0.9967060462154962, 0.9877697391019615,
                          0.98401636296503, 0.9905921230630785, 0.990570758079667, 0.9847896602829518, 0.9842767938040367,
                          0.9516618302498175, 0.9894203625002091, 0.9713330344039811, 0.9448128823732177, 0.9544919165416704]
    e_joint = pascal_e_joint if cfg['dataset'] == 'pascal' else cityscapes_e_joint
    return e_joint
