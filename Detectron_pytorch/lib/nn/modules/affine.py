import torch
import torch.nn as nn
import torch.nn.functional as F

# FrozenBN in detectron2
class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

#class AffineChannel2d(nn.Module):
#    """ A simple channel-wise affine transformation operation """
#    def __init__(self, num_features):
#        super().__init__()
#        self.num_features = num_features
#        self.weight = nn.Parameter(torch.Tensor(num_features))
#        self.bias = nn.Parameter(torch.Tensor(num_features))
#        self.weight.data.uniform_()
#        self.bias.data.zero_()
#
#    def forward(self, x):
#        return x * self.weight.view(1, self.num_features, 1, 1) + \
#            self.bias.view(1, self.num_features, 1, 1)
#