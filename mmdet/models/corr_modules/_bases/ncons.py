# 20200904 NCNet related
import torch
import torch.nn as nn
from .conv4d import Conv4d
from .features import featureL2Norm

class FeatureCorrelation(torch.nn.Module):
    def __init__(self, normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):

        b, c, hA, wA = feature_A.size()
        b, c, hB, wB = feature_B.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # size [b,c,h*w]
        feature_B = feature_B.view(b, c, hB * wB)  # size [b,c,h*w]
        # perform matrix mult.
        feature_mul = torch.bmm(feature_A, feature_B)
        # indexed [batch,row_A,col_A,row_B,col_B]
        correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(1)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor

# 20200818 update, copy code from ncnet repo
class NeighConsensus(torch.nn.Module):
    def __init__(
        self,
        #use_cuda=True,
        kernel_sizes=[5, 5, 5],
        channels=[1, 16, 16, 1],
        symmetric_mode=True,
    ):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i == 0:
                ch_in = 1
            else:
                ch_in = channels[i - 1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        # if use_cuda:
        #     self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x

# copy from ANCNet-CVPR2020
def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (
        corr4d_A * corr4d_B
    )  # parenthesis are important for symmetric output

    return corr4d

if __name__ == '__main__':
    print()