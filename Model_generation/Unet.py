import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ContinuousConditionalBatchNorm2d(nn.Module):
    def __init__(self, n_channels, input_dim):
        super().__init__()
        self.n_channels = n_channels
        #BatchNorm with affine=False is just normalization without params
        self.bn = nn.BatchNorm2d(n_channels, affine=False)
        #Map continuous condition to required size
        self.linear = nn.Linear(input_dim, n_channels*2)

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.linear(y).chunk(2, 1)
        out = gamma.view(-1, self.n_channels, 1, 1) * out + beta.view(-1, self.n_channels, 1, 1)
        return out


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, n_channels, num_classes):
        super().__init__()
        self.n_channels = n_channels
        #BatchNorm with affine=False is just normalization without params
        self.bn = nn.BatchNorm2d(n_channels, affine=False)
        self.embed = nn.Embedding(num_classes, n_channels * 2)
        #First half of embedding is for gamma (scale parameter)
        self.embed.weight.data[:, :n_channels].normal_(1, 0.02)
        #Second half of the embedding is for beta (bias parameter)
        self.embed.weight.data[:, n_channels:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.n_channels, 1, 1) * out + beta.view(-1, self.n_channels, 1, 1)
        return out

class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d,self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)

class MultipleConditionalBatchNorm2d(nn.Module):
    """Conditional BatchNorm to be used in the case of multiple categorical input features.
    Args:
      - n_channels (int): number of feature maps of the convolutional layer to be conditioned
      - num_classes (iterable of ints): list of the number of classes for the different feature embeddings
      - adaptive (bool): use adaptive batchnorm instead of standadrd batchnorm
    """
    def __init__(self, n_channels, nums_classes, adaptive=False):
        super().__init__()
        self.n_channels = n_channels
        self.nums_classes = nums_classes
        #use embedding dim such that the total size is n_channels
        embedding_dims = [n_channels//len(nums_classes) for i in range(len(nums_classes)-1)]
        embedding_dims.append((n_channels//len(nums_classes)) + (n_channels % len(nums_classes)))
        embedding_dims = [dim*2 for dim in embedding_dims]

        #BatchNorm with affine=False is just normalization without params
        self.bn = AdaptiveBatchNorm2d(n_channels, affine=False) if adaptive else nn.BatchNorm2d(n_channels, affine=False)
        #An embedding for each different categorical feature
        self.embeddings = nn.ModuleList([nn.Embedding(num_classes, dim) for num_classes, dim in zip(nums_classes, embedding_dims)])

        #Initialize embeddings
        for emb in self.embeddings:
          #First half of embedding is for gamma (scale parameter)
          emb.weight.data[:, :n_channels].normal_(1, 0.02)
          #Second half of the embedding is for beta (bias parameter)
          emb.weight.data[:, n_channels:].zero_()

    def forward(self, x, class_idxs):
        out = self.bn(x)
        concatenated_embeddings = [emb(idx) for emb, idx in zip(self.embeddings, class_idxs)]
        concatenated_embeddings = torch.cat(concatenated_embeddings, dim=1)
        #gamma, beta = concatenated_embeddings.chunk(2, 1)
        gamma, beta = concatenated_embeddings[:,::2], concatenated_embeddings[:,1::2]
        #print(gamma, beta)
        out = gamma.view(-1, self.n_channels, 1, 1) * out + beta.view(-1, self.n_channels, 1, 1)
        return out


class MLP(nn.Module):
    """
    A one-hidden-layer MLP.
    Takes (batch_size, 1, imsize, imsize) tensors as input, and outputs tensor of same shape.
    Outputs in range [-1,1]
    """
    def __init__(self, imsize=32, n_channels=1, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.imsize = imsize
        self.net = nn.Sequential(
            nn.Linear(imsize*imsize*n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, imsize*imsize*n_channels),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, n_channels = x.size(0), x.size(1)
        x = x.view(batch_size, -1)
        x = self.net(x)
        x = x.view(batch_size, n_channels, self.imsize, self.imsize)
        return x



class UNet(nn.Module):
    """
      Standard Unet
    """
    def __init__(self):
        super().__init__()
        self.inc = unet_block(3,64,False)
        self.down1 = unet_block(64,128)
        self.down2 = unet_block(128,256)
        self.down3 = unet_block(256,512)
        self.down4 = unet_block(512,512)
        self.up1 = unet_up(1024,256)
        self.up2 = unet_up(512,128)
        self.up3 = unet_up(256,64)
        self.up4 = unet_up(128,64)
        self.outc = nn.Conv2d(64,3,1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class unet_block(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self,in_ch,out_ch,down=True):
        super(unet_block,self).__init__()
        self.down = down
        self.pool = nn.MaxPool2d(2)
        self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True) )

    def forward(self,x):
        if self.down:
            x = self.pool(x)
        x = self.block(x)
        return x

class unet_up(nn.Module):
    def __init__(self,in_ch,out_ch,bilinear=True):
        super(unet_up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Conv2dTranspose(in_ch//2,out_ch//2,2,stride=2)

        self.conv = unet_block(in_ch,out_ch,False)
    
    def forward(self,x1,x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x        


class ConditionalUNet(nn.Module):
    """
      Conditional Unet
    """
    def __init__(self, nums_classes=(6,3,3,4)):
        super().__init__()
        self.inc = unet_block(3,64,False)
        self.down1 = unet_block(64,128)
        self.down2 = unet_block(128,256)
        self.down3 = unet_block(256,512)
        self.down4 = unet_block(512,512)
        self.up1 = cond_unet_up(1024,256, nums_classes=nums_classes)
        self.up2 = cond_unet_up(512,128, nums_classes=nums_classes)
        self.up3 = cond_unet_up(256,64, nums_classes=nums_classes)
        self.up4 = cond_unet_up(128,64, nums_classes=nums_classes)
        self.outc = nn.Conv2d(64,3,1)

    def forward(self, x, feat):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4, feat)
        x = self.up2(x, x3, feat)
        x = self.up3(x, x2, feat)
        x = self.up4(x, x1, feat)
        x = self.outc(x)
        return x

class cond_unet_block(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self,in_ch,out_ch,down=True, nums_classes=(6,3,3,4)):
        super().__init__()
        self.down = down
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = MultipleConditionalBatchNorm2d(out_ch, nums_classes)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = MultipleConditionalBatchNorm2d(out_ch, nums_classes)

    def forward(self,x,feat):
        if self.down:
            x = self.pool(x)
        #x = self.block(x)
        x = self.conv1(x)
        x = F.relu(self.bn1(x,feat))
        x = self.conv2(x)
        x = F.relu(self.bn2(x,feat))
        return x

class cond_unet_up(nn.Module):
    def __init__(self,in_ch,out_ch,bilinear=True,nums_classes=(6,3,3,4)):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Conv2dTranspose(in_ch//2,out_ch//2,2,stride=2)

        self.conv = cond_unet_block(in_ch,out_ch,False,nums_classes)

    def forward(self,x1,x2,feat):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x,feat)
        return x



class LittleUnet(nn.Module):
    """A little U-net style CNN based on concatenations and transposed convolution
    Output in [-1,1]
    """
    def __init__(self, imsize=32, n_channels=1, initial_1by1=False):
        super().__init__()
        if initial_1by1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, 255, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(255, 255, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(255, 32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(n_channels, 32, kernel_size=3, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_tran1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_tran2 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_tran3 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 1, kernel_size=4, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        x = self.conv_tran1(out3)
        x = self.conv_tran2(torch.cat((out2, x), dim=1))
        x = self.conv_tran3(torch.cat((out1, x), dim=1))
        return x


class VGG(nn.Module):
    def __init__(self,device):
        super(VGG,self).__init__()
        self.model = vgg16_bn(True).features
        self.mean = torch.Tensor([123.68,  116.779,  103.939]).view(1,3,1,1)
        self.mean = self.mean.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Normalize the images since we have [-1,1] and vgg wants [0,1]
        x = (x*0.5)+0.5
        x = x*255 - self.mean
        x = self.model(x)
        return x

if __name__ == "__main__":
    im = torch.randn(8, 1, 32, 32)

    mlp = MLP()
    #Test mlp forward
    assert mlp(im).size() == im.size()


    
    unet = LittleUnet(initial_1by1=True)
    #Test little unet forward
    assert unet(im).size() == im.size()
    
    im = torch.randn(8, 3, 60, 80)

    
    feat = torch.randn(1, 32, 300, 500)
    class_idxs = (torch.LongTensor([1]), torch.LongTensor([0]), torch.LongTensor([0]), torch.LongTensor([1]))
    nums_classes = (6, 3, 3, 4)
    cbn = MultipleConditionalBatchNorm2d(n_channels=32, nums_classes=nums_classes)
    assert cbn(feat, class_idxs).size() == feat.size()
    


    
    im = torch.randn(1, 3, 300, 500)
    unet = UNet()
    assert unet(im).size() == im.size()
    print("Tests run correctly!")
