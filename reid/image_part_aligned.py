import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.googlenet import GoogLeNet


class Model(nn.Module):
    def __init__(self, n_parts=8):
        super(Model, self).__init__()
        self.n_parts = n_parts

        self.feat_conv = GoogLeNet()
        self.conv_input_feat = nn.Conv2d(self.feat_conv.output_channels, 512, 1)

        # part net
        self.conv_att = nn.Conv2d(512, self.n_parts, 1)

        for i in range(self.n_parts):
            setattr(self, 'linear_feature{}'.format(i+1), nn.Linear(512, 64))

    def forward(self, x):
        feature = self.feat_conv(x)
        feature = self.conv_input_feat(feature)

        att_weights = torch.sigmoid(self.conv_att(feature))

        linear_feautres = []
        for i in range(self.n_parts):
            masked_feature = feature * torch.unsqueeze(att_weights[:, i], 1)
            pooled_feature = F.avg_pool2d(masked_feature, masked_feature.size()[2:4])
            linear_feautres.append(
                getattr(self, 'linear_feature{}'.format(i+1))(pooled_feature.view(pooled_feature.size(0), -1))
            )

        concat_features = torch.cat(linear_feautres, 1)
        normed_feature = concat_features / torch.clamp(torch.norm(concat_features, 2, 1, keepdim=True), min=1e-6)

        return normed_feature
