import cv2
import numpy as np
import torch

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def get_expectation(trani_loader,classes = 19):
    #or 21
    counter,totals = [0]*classes,[0]*classes
    loader_iter = iter(trani_loader)
    for step in range(len(trani_loader)):
        _, label = loader_iter.next()
        batch_size, h, w = label.size()
        #outs = model(image)
        #pred, rep = outs["pred"], outs["rep"]
        #pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)
        ext = torch.nn.functional.pad(label, (1, 1, 1, 1, 0, 0))
        left = ext[:,:-2, 1:-1]
        right = ext[:,2:,1:-1]
        up = ext[:,1:-1,:-2]
        down = ext[:,1:-1, 2:]
        d=torch.stack((left, right, up, down))
        def expand(l,classes=19):
            if len(l) !=classes:
                return l+[0]*(classes-len(l))
            return l
        _totals = expand(list(label.reshape(-1).bincount().numpy()))
        totals = [a + b for a, b in zip(totals, _totals)]
        for neigbor in d:
            inds=torch.where(neigbor==label)
            _counter = expand(list(label[inds].bincount().numpy()))
            counter = [a + b for a, b in zip(counter, _counter)]
    probs = [a/(4*b) for a,b in zip(counter,totals)]
    print(probs)
    return probs

def neighbor_by_factor(arr: torch.Tensor, factor: int):
    ext = torch.nn.functional.pad(torch.clone(arr), (1, 1, 1, 1, 0, 0, 0, 0))
    left = ext[:, :, :-factor*2, factor:-factor]
    right = ext[:, :, factor*2:, factor:-factor]
    up = ext[:, :, factor:-factor, :-factor*2]
    down = ext[:, :, factor:-factor, factor*2:]
    d = torch.stack((left, right, up, down)).to(arr.device)
    arr, neighbor_idx = torch.max(d, 0)
    return arr, neighbor_idx