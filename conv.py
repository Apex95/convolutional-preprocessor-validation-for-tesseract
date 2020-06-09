import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def convolve(input):
  
    global kernels

    output = input

    # convolve
    for i in range(n_of_kernels):
        #print(1+i)
        output = torch.nn.functional.conv2d(output, kernels[i], bias=None, padding=1, stride=1, groups=n_of_channels) #.to(device)
        output = torch.nn.functional.relu(output)


    batch_of_images = output.permute(0,2,3,1)
    batch_of_images = torch.clamp(batch_of_images, 0, 255).type(torch.uint8).cpu().numpy()

    return batch_of_images



def reset_kernels():
    global kernels
    
    kernels = []

    for i in range(n_of_kernels):
      kernels.append(torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]).view(1, 1, kernels_sizes[i], kernels_sizes[i]).repeat(n_of_channels, 1, 1, 1))

    
class ConvProcessor(torch.nn.Module):
    def __init__(self):
        super(ConvProcessor, self).__init__()
        
        # 1x1x3 -> 3x3x1
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        #torch.nn.init.constant_(self.conv1.weight, 0)

        # horizontal symmetry
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        #torch.nn.init.constant_(self.conv2.weight, 0)
        
        # vertical symmetry
        self.conv3 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        #torch.nn.init.constant_(self.conv3.weight, 0)

        # first diagonal
        self.conv4 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # second diagonal
        self.conv5 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = x.permute(0,2,3,1) # * 255
        x = torch.clamp(x, 0, 255).type(torch.uint8).cpu().numpy()

        return x

    def compress_weights(self, weights):

        # conv2
        weights = torch.cat([weights[0:9], weights[12:]])
        
        # conv3
        weights = torch.cat([weights[0:11], weights[12:]])
        weights = torch.cat([weights[0:13], weights[14:]])
        weights = torch.cat([weights[0:15], weights[16:]])

        # conv4
        weights = torch.cat([weights[0:16], weights[18:]])
        weights = torch.cat([weights[0:18], weights[19:]])

        # conv5
        weights = torch.cat([weights[0:26], weights[27:]])
        weights = weights[0:27]

        return weights


    def uncompress_weights(self, weights):

        # conv2
        weights = torch.cat([weights[0:9], weights[3:6], weights[9:]])
        
        # conv3
        weights = torch.cat([weights[0:14], weights[12:13], weights[14:]])
        weights = torch.cat([weights[0:17], weights[15:16], weights[17:]])
        weights = torch.cat([weights[0:20], weights[18:19], weights[20:]])
        

        # conv 4
        weights = torch.cat([weights[0:23], weights[24:25], weights[22:]])
        weights = torch.cat([weights[0:26], weights[27:28], weights[26:]])
        weights = torch.cat([weights[0:29], weights[29:30], weights[29:]])

        # conv 5
        weights = torch.cat([weights[0:30], weights[31:36], weights[32:33], weights[36:]])
        weights = torch.cat([weights[0:37], weights[33:34], weights[30:31]])

        return weights


    def get_weights(self):

        weights = []
        self.sizes = {}
        self.shapes = {}

        for name, param in self.named_parameters():
            if len(weights) == 0:
                _param = param.flatten()
                weights = param.flatten()

                self.sizes[name] = len(_param)
                self.shapes[name] = param.shape
            else:
                _param = param.flatten()
                weights = torch.cat((weights, _param), 0)

                self.sizes[name] = len(_param)
                self.shapes[name] = param.shape

        weights = self.compress_weights(weights)
        

        return weights

    def set_weights(self, weights):

        weights = self.uncompress_weights(weights)

        state_dict = self.state_dict()

        for name, param in state_dict.items():
            
            size = self.sizes[name]
            shape = self.shapes[name]

            new_value = weights[:size]
            weights = weights[size:]

            state_dict[name].copy_(new_value.view(shape))


    def reset_weights(self):
        identity_kernels = torch.Tensor(list([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
        
        self.set_weights(self.compress_weights(identity_kernels))
