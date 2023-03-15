import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import save_image
import os.path as osp



def visual_evaluation(pred_u_large,golden_label_u,pred_u_teacher,drop_percent,path='imgs'):
    pascal_E_joint = [0.9980886695132567, 0.9860495622941258, 0.9555923573386919, 0.9888166500147213,
     0.9861891841249937, 0.9894536447668415, 0.9928481727772317, 0.9899944432748093, 0.9939554498312092,
      0.9723734804881907, 0.9814734877790432, 0.992167780556063, 0.9921082109870506, 0.9904260573555301,
       0.9877020724598903, 0.9890704556150036, 0.9771896568239452, 0.9865771364049042, 0.9878175644560833,
        0.9948415228771397, 0.993315147276186]
    batch_size, num_class, h, w = pred_u_teacher.shape
    target= golden_label_u
    
    print(f'golden label shape: {target.shape}, teacher pred shape: {pred_u_teacher.shape}')
    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_u_teacher, dim=1) #shape: batch_size, num_class, h, w
        max_probs, max_class = torch.max(prob, dim=1) #shape: batch_size, h, w

        #entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1) #shape: batch_size, h, w
        #thresh = np.percentile(entropy[target != 255].detach().cpu().numpy().flatten(), drop_percent)
        #thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool() #shape: batch_size, h, w


        ext = torch.nn.functional.pad(torch.clone(prob), (1,1,1,1,0,0,0,0))
        left = ext[:,:,:-2, 1:-1]
        right = ext[:,:,2:,1:-1]
        up = ext[:,:,1:-1,:-2]
        down = ext[:,:,1:-1, 2:]

        d=torch.stack((left, right, up, down))
        arr=torch.max(d,0)[0]
        prob = prob + arr - prob*arr
                
        entropy_s4n = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        thresh_mask_s4n = entropy_s4n.ge(thresh).bool() * (target != 255).bool()
        
        mask_pos = torch.ones_like(target)
        mask_pos[thresh_mask_s4n] = 0

        max_class[thresh_mask_s4n] = 0
        golden_label_u = golden_label_u * (target == max_class)
        print(golden_label_u.shape)
        for mask_in in en_label_u:
            save_image(mask_in.float(), osp.hoint(path,'label1.png'))
            #save the tensor to image:
            assert 1==0 




        #mask_out = thresh_mask_s4n!=thresh_mask
        #only_ours = thresh_mask_s4n*mask_out
