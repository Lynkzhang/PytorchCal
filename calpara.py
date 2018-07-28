#This calculted conv parameters in model.
#This script only calculate conv layer filters now.

import torch

net=torch.load("checkpoint_latest.pth.tar")
para=net['state_dict']
Totalpara=0
for i in para:
        layer=i.split(".")
        if layer[-1]=='weight':
                if layer[-2][:4] == 'conv':
                        print("Layer of %s have a shape of:"%(i))
                        print(para[i].size())
                        shape_layer=para[i].size()
                        para_layer=shape_layer[0]*shape_layer[1]*shape_layer[2]*shape_layer[3]
                        print("This layer has %d parameters"%(para_layer))
                        Totalpara+=para_layer

print("Total para is:")
print(Totalpara)
