import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import math
import torch.utils.model_zoo as model_zoo


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


class Reconstructor_DRAVL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Reconstructor_DRAVL, self).__init__()
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, output_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.dense(x)


class GlobalPreModel_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GlobalPreModel_NN, self).__init__()
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 300),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Linear(300, 200),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Linear(200, 100),
            #nn.Dropout(),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        return self.dense(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 600), 
            nn.LayerNorm(600),
            nn.ReLU(),
            
            nn.Linear(600, 200), 
            nn.LayerNorm(200),
            nn.ReLU(),
            
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            
            nn.Linear(100, target_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class MID_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MID_layer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, input_dim*5, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim*5, output_dim, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        return x


class MID_enlarge_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MID_enlarge_layer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        # self.layer1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(input_dim, input_dim*10, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(input_dim*10, input_dim*4, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(input_dim*4, output_dim, bias=True),
        #     nn.ReLU(inplace=True)
        # )
    
    def forward(self, x):
        x = self.layer1(x)
        return x


class ActivePartySplitNN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*2)
        self.classifier_head = nn.Linear(dim*2, dim)

    def forward(self, x):
        return self.classifier_head(self.fc1(x))


class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32, bias=True),
            nn.ReLU(inplace=True)
        )
        torch.nn.init.xavier_uniform_(self.layer1[1].weight)
        torch.nn.init.zeros_(self.layer1[1].bias)

        self.layer2 = nn.Sequential(
            nn.Linear(32, output_dim, bias=True),
            #nn.ReLU(inplace=True)
        )
        #torch.nn.init.xavier_uniform_(self.layer2[0].weight)
        #torch.nn.init.zeros_(self.layer2[0].bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MLP2_middle_leak(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2_middle_leak, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32, bias=True),
            nn.ReLU(inplace=True)
        )
        torch.nn.init.xavier_uniform_(self.layer1[1].weight)
        torch.nn.init.zeros_(self.layer1[1].bias)

        self.layer2 = nn.Sequential(
            nn.Linear(32, output_dim, bias=True),
            #nn.ReLU(inplace=True)
        )
        #torch.nn.init.xavier_uniform_(self.layer2[0].weight)
        #torch.nn.init.zeros_(self.layer2[0].bias)

    def forward(self, x):
        middle_input = self.layer1(x)
        x = self.layer2(middle_input)
        return middle_input, x

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        # act = nn.Tanh
        act = nn.ReLU
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=60, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=480, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 20, 5)
        self.fc1 = nn.Linear(20 * 1 * 5, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 1 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        return x


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         act = nn.Sigmoid
#         self.body = nn.Sequential(
#             nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
#             act(),
#             nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
#             act(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(768, 100)
#         )
#
#     def forward(self, x):
#         out = self.body(x)
#         out = out.view(out.size(0), -1)
#         # print(out.size())
#         out = self.fc(out)
#         return out
#
#
# class LeNet2(nn.Module):
#     def __init__(self, classes = 2):
#         super(LeNet2, self).__init__()
#         act = nn.Sigmoid
#         self.body = nn.Sequential(
#             nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
#             act(), # 16 * 8 * 12
#             nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
#             act(), # 8 * 4 * 12
#             nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
#             act(), # 8 * 4 * 12
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(384, classes)
#         )
#
#     def forward(self, x):
#         out = self.body(x)
#         out = out.view(out.size(0), -1)
#         # print(out.size())
#         out = self.fc(out)
#         return out
#
#
class LeNet3(nn.Module):
    def __init__(self, classes=2):
        super(LeNet3, self).__init__()
        # act = nn.Sigmoid
        act = nn.LeakyReLU
        # act = nn.ReLU
        padding_1 = 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3, padding=padding_1, stride=1),
            act(),  # 128 * 64 * 12
            nn.Conv2d(in_channels=18, out_channels=36, kernel_size=3, padding=padding_1, stride=1),
            act(),  # 64 * 32 * 12
            nn.Conv2d(in_channels=36, out_channels=64, kernel_size=3, stride=1),
            act(),  # 64 * 32 * 12
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            act(),  # 64 * 32 * 12
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc = nn.Sequential(
            # nn.Linear(64 * 16 * 16, classes)
            # nn.Linear(25088, classes)
            nn.Linear(10752, classes)

        )

    def forward(self, x):
        out = self.body(x)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        # print("out: ", out.size())
        out = self.fc(out)
        return out

class ActivePartyWithoutTrainableLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_a, pred_b):
        pred = pred_a + pred_b
        return pred


class ActivePartyWithTrainableLayer(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, pred_a, pred_b):
        out = torch.cat([pred_a, pred_b], dim=1) # out = [dim0, dim1_a+dim1_b], dim0 is number of samples
        return self.classifier_head(out)

    # def get_prediction(self, z0, z_list):
    #     if z_list is not None:
    #         out = torch.cat([z0] + z_list, dim=1)
    #     else:
    #         out = z0
    #     return self.classifier_head(out)


class ActivePartyWithTrainableLayer_catinated(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # out = torch.cat([pred_a, pred_b], dim=1) # out = [dim0, dim1_a+dim1_b], dim0 is number of samples
        return self.classifier_head(x)

    # def get_prediction(self, z0, z_list):
    #     if z_list is not None:
    #         out = torch.cat([z0] + z_list, dim=1)
    #     else:
    #         out = z0
    #     return self.classifier_head(out)



class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 2
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.dropout(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class ResNet_middel_leak(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.dropout(output)
        middle_input = output.view(output.size(0), -1)
        output = self.fc(middle_input)

        return middle_input, output


def resnet18_middle_leak(num_classes):
    """ return a ResNet 18 object
    """
    return ResNet_middel_leak(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet18(num_classes):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes)

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])
