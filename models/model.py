import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskNet(nn.Module):
    def __init__(self, num_labels, gpu_device=0, use_class=False):
        super(MultiTaskNet, self).__init__()
        # Load pre-trained VGG-16 weights to two separate variables.
        # They will be used in defining the depth and RGB encoder sequential layers.
        feats = list(models.vgg16(pretrained=True).features.children())
        feats2 = list(models.vgg16(pretrained=True).features.children())

        # Average the first layer of feats variable, the input-layer weights of VGG-16,
        # over the channel dimension, as depth encoder will be accepting one-dimensional
        # inputs instead of three.
        avg = torch.mean(feats[0].cuda(gpu_device).weight.data, dim=1)
        avg = avg.unsqueeze(1)

        bn_moment = 0.1
        self.use_class = use_class

        if use_class:
            num_classes = 10
       
        # RGB ENCODER
        self.CBR1_RGB = nn.Sequential(
            feats2[0].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats2[1].cuda(gpu_device),
            feats2[2].cuda(gpu_device),
            nn.BatchNorm2d(64).cuda(gpu_device),
            feats2[3].cuda(gpu_device),
        )

        self.CBR2_RGB = nn.Sequential(
            feats2[5].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats2[6].cuda(gpu_device),
            feats2[7].cuda(gpu_device),
            nn.BatchNorm2d(128).cuda(gpu_device),
            feats2[8].cuda(gpu_device),
        )

        self.CBR3_RGB = nn.Sequential(
            feats2[10].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[11].cuda(gpu_device),
            feats2[12].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[13].cuda(gpu_device),
            feats2[14].cuda(gpu_device),
            nn.BatchNorm2d(256).cuda(gpu_device),
            feats2[15].cuda(gpu_device),
        )
        self.dropout3 = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR4_RGB = nn.Sequential(
            feats2[17].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[18].cuda(gpu_device),
            feats2[19].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[20].cuda(gpu_device),
            feats2[21].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[22].cuda(gpu_device),
        )
        self.dropout4 = nn.Dropout(p=0.5).cuda(gpu_device)

        self.CBR5_RGB = nn.Sequential(
            feats2[24].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[25].cuda(gpu_device),
            feats2[26].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[27].cuda(gpu_device),
            feats2[28].cuda(gpu_device),
            nn.BatchNorm2d(512).cuda(gpu_device),
            feats2[29].cuda(gpu_device),
        )
        self.dropout5 = nn.Dropout(p=0.5).cuda(gpu_device)

        if use_class:
            self.ClassHead = nn.Sequential(
                # classifier[0].cuda(gpu_device),
                nn.Linear(35840, 4096).cuda(gpu_device),
                nn.ReLU(),
                nn.Dropout(p=0.5).cuda(gpu_device),
                nn.Linear(4096, 4096).cuda(gpu_device),
                # classifier[3].cuda(gpu_device),
                nn.ReLU(),
                nn.Dropout(p=0.5).cuda(gpu_device),
                nn.Linear(4096, num_classes).cuda(gpu_device)
            )

        # SEMANTIC DECODER
        self.CBR5_Dec = nn.Sequential(
            nn.Conv2d(512+512, 512+512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512+512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512+512, 512+512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512+512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512+512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR4_Dec = nn.Sequential(
            nn.Conv2d(512+512, 512+512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512+512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512+512, 512+512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512+512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512+512, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR3_Dec = nn.Sequential(
            nn.Conv2d(256+256, 256+256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256+256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(256+256, 256+256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256+256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(256+256,  128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR2_Dec = nn.Sequential(
            nn.Conv2d(128+128, 128+128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128+128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(128+128, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )

        self.CBR1_Dec = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(64, num_labels, kernel_size=3, padding=1).cuda(gpu_device),
        )


	# DEPTH DECODER
        self.CBR5_Depth_Dec = nn.Sequential(
            nn.Conv2d(512+512, 512+512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512+512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512+512, 512+512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512+512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512+512, 512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR4_Depth_Dec = nn.Sequential(
            nn.Conv2d(512+512, 512+512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512+512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512+512, 512+512, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(512+512, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(512+512, 256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR3_Depth_Dec = nn.Sequential(
            nn.Conv2d(256+256, 256+256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256+256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(256+256, 256+256, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(256+256, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(256+256,  128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Dropout(p=0.5).cuda(gpu_device),
        )

        self.CBR2_Depth_Dec = nn.Sequential(
            nn.Conv2d(128+128, 128+128, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(128+128, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(128+128, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
        )

        self.CBR1_Depth_Dec = nn.Sequential(
            nn.Conv2d(64+64, 64, kernel_size=3, padding=1).cuda(gpu_device),
            nn.BatchNorm2d(64, momentum=bn_moment).cuda(gpu_device),
            nn.ReLU().cuda(gpu_device),
            nn.Conv2d(64, 1, kernel_size=3, padding=1).cuda(gpu_device),
        )


        print('[INFO] MultiTaskNet model has been created')
        self.initialize_weights()

    # He Initialization for the linear layers in the classification head
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_out = size[0]  # number of rows
                fan_in = size[1]  # number of columns
                variance = np.sqrt(4.0/(fan_in + fan_out))
                m.weight.data.normal_(0.0, variance)

    def forward(self, rgb_inputs):

        # RGB ENCODER
        # Stage 1
        y_1 = self.CBR1_RGB(rgb_inputs)
        y, id1 = F.max_pool2d(y_1, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        y_2 = self.CBR2_RGB(y)
        y, id2 = F.max_pool2d(y_2, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        y_3 = self.CBR3_RGB(y)
        y, id3 = F.max_pool2d(y_3, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout3(y)

        # Stage 4
        y_4 = self.CBR4_RGB(y)
        y, id4 = F.max_pool2d(y_4, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout4(y)

        # Stage 5
        y_5 = self.CBR5_RGB(y)
        y_size = y.size()

        y, id5 = F.max_pool2d(y_5, kernel_size=2, stride=2, return_indices=True)
        y = self.dropout5(y)

        if self.use_class:
            # FC Block for Scene Classification
            y_class = y.view(y.size(0), -1)
            y_class = self.ClassHead(y_class)

        # SEMANTIC DECODER
        # Stage 5 dec
        s = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)        
        s = torch.cat([s, y_5], 1)
        s = self.CBR5_Dec(s)

        # Stage 4 dec
        s = F.max_unpool2d(s, id4, kernel_size=2, stride=2)
        s = torch.cat([s, y_4], 1)
        s = self.CBR4_Dec(s)

        # Stage 3 dec
        s = F.max_unpool2d(s, id3, kernel_size=2, stride=2)
        s = torch.cat([s, y_3], 1)
        s = self.CBR3_Dec(s)

        # Stage 2 dec
        s = F.max_unpool2d(s, id2, kernel_size=2, stride=2)
        s = torch.cat([s, y_2], 1)
        s = self.CBR2_Dec(s)

        # Stage 1 dec
        s = F.max_unpool2d(s, id1, kernel_size=2, stride=2)
        s = torch.cat([s, y_1], 1)
        s = self.CBR1_Dec(s)


        # DEPTH DECODER
        # Stage 5 dec
        d = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
        d = torch.cat([d, y_5], 1)
        d = self.CBR5_Depth_Dec(d)

        # Stage 4 dec
        d = F.max_unpool2d(d, id4, kernel_size=2, stride=2)
        d = torch.cat([d, y_4], 1)
        d = self.CBR4_Depth_Dec(d)

        # Stage 3 dec
        d = F.max_unpool2d(d, id3, kernel_size=2, stride=2)
        d = torch.cat([d, y_3], 1)
        d = self.CBR3_Depth_Dec(d)

        # Stage 2 dec
        d = F.max_unpool2d(d, id2, kernel_size=2, stride=2)
        d = torch.cat([d, y_2], 1)
        d = self.CBR2_Depth_Dec(d)

        # Stage 1 dec
        d = F.max_unpool2d(d, id1, kernel_size=2, stride=2)
        d = torch.cat([d, y_1], 1)
        d = self.CBR1_Depth_Dec(d)

        if self.use_class:
            return s, y_class, d
        return s, d
