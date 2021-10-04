"""Provides functions that manipulate boxes and points"""
# hsy Transfer key-points using bilinear interpolation
# Note: I use framework of DHPF but adapt to eval protocal of NCNet.
# DHPF from src to trg, while NCNet (and here) is trg to src.

import torch
import numpy as np
from torch.autograd import Variable

class Geometry:
    @classmethod
    def initialize(cls, feat_size, device, do_softmax):
        cls.max_pts = 400
        cls.eps = 1e-30
        cls.rfs = cls.receptive_fields(811, 16, feat_size).to(device) # RF for res101_conv4-23 811, jump 16.0
        cls.rf_center = Geometry.center(cls.rfs)
        assert (do_softmax in [True, False])
        cls.do_softmax = do_softmax

    @classmethod
    def receptive_fields(cls, rfsz, jsz, feat_size):
        r"""Returns a set of receptive fields (N, 4)"""
        width = feat_size[1]
        height = feat_size[0]

        feat_ids = torch.tensor(list(range(width))).repeat(1, height).t().repeat(1, 2)
        feat_ids[:, 0] = torch.tensor(list(range(height))).unsqueeze(1).repeat(1, width).view(-1)

        box = torch.zeros(feat_ids.size()[0], 4)
        box[:, 0] = feat_ids[:, 1] * jsz - rfsz // 2
        box[:, 1] = feat_ids[:, 0] * jsz - rfsz // 2
        box[:, 2] = feat_ids[:, 1] * jsz + rfsz // 2
        box[:, 3] = feat_ids[:, 0] * jsz + rfsz // 2

        return box

    @classmethod
    def center(cls, box):
        r"""Computes centers, (x, y), of box (N, 4)"""
        x_center = box[:, 0] + (box[:, 2] - box[:, 0]) // 2
        y_center = box[:, 1] + (box[:, 3] - box[:, 1]) // 2
        return torch.stack((x_center, y_center)).t().to(box.device)

    @classmethod
    def gaussian2d(cls, side=7):
        r"""Returns 2-dimensional gaussian filter"""
        dim = [side, side]

        siz = torch.LongTensor(dim)
        sig_sq = (siz.float() / 2 / 2.354).pow(2)
        siz2 = (siz - 1) / 2

        x_axis = torch.arange(-siz2[0], siz2[0] + 1).unsqueeze(0).expand(dim).float()
        y_axis = torch.arange(-siz2[1], siz2[1] + 1).unsqueeze(1).expand(dim).float()

        gaussian = torch.exp(-(x_axis.pow(2) / 2 / sig_sq[0] + y_axis.pow(2) / 2 / sig_sq[1]))
        gaussian = gaussian / gaussian.sum()

        return gaussian

    @classmethod
    def transfer_kps(cls, corr4d, trg_kps, n_pts, img_side):
        r"""Transfer keypoints by nearest-neighbour assignment"""
        # corr4d: (b,c,hs,ws,ht,wt), where c==1
        # transfer kps in trg img to coordinates in src img
        # trg_kps: Bx2xN

        b, c, hs, ws, ht, wt = corr4d.shape
        assert (c == 1)
        device = corr4d.device
        src_imsizes = img_side * torch.ones((b, 2)).to(device)
        trg_imsizes = img_side * torch.ones((b, 2)).to(device)

        prd_kps = []
        for ct, kpss, npt, _src_imsize, _trg_imsize in zip(corr4d, trg_kps, n_pts, src_imsizes, trg_imsizes):
            # 0. prepare
            kp = kpss.narrow_copy(1, 0, npt).unsqueeze(0)  # B x 2 x N ,w.r.t. the original image coordinate #240
            ct = ct.unsqueeze(0)
            _src_imsize = _src_imsize.unsqueeze(0)
            _trg_imsize = _trg_imsize.unsqueeze(0)
            assert (ct.shape == (1, c, hs, ws, ht, wt))
            assert (_src_imsize.shape == _trg_imsize.shape == (1, 2))

            # 1. get matches
            # as interv_direct = True, xA, yA of src img is meshgrid; #todo whether invert direction
            xA, yA, xB, yB, sB = cls.corr_to_matches(ct, do_softmax=True, invert_matching_direction=False)
            matches = (xA, yA, xB, yB)

            # 2. convert from image coordinate to -1, 1. _src_imsize: [B,2], where dim0@bsz and B==1, dim1@h,w
            key_points_norm = cls.PointsToUnitCoords(kp, _trg_imsize)  # B x 2 x N

            # 3. transfer normalized key points with estimated transformations (transfer kps in src img to trg img)
            warped_points_norm = cls.bilinearInterpPointTnf(matches, key_points_norm)  # B x 2 x N

            # 4. revert back to PixelCoords
            warped_points = cls.PointsToPixelCoords(warped_points_norm, _src_imsize)  # B x 2 x N
            prd = warped_points.squeeze(0)
            assert (prd.shape == (2, npt))

            # 5. Concatenate pad-points
            pads = (torch.zeros((2, cls.max_pts - npt)).to(prd.device) - 1)
            prd = torch.cat([prd, pads], dim=1)
            prd_kps.append(prd)

        return torch.stack(prd_kps)

    @classmethod
    def unnormalize_axis(cls, x, L):
        return x * (L - 1) / 2 + 1 + (L - 1) / 2

    @classmethod
    def normalize_axis(cls, x, L):
        # convert original pixel coordinate into unit coordinate (-1, 1)
        return (x - 1 - (L - 1) / 2) * 2 / (L - 1)
        # return (x-L/2)*2/L

    @classmethod
    def PointsToPixelCoords(cls, P, im_size):
        h, w = im_size[:, 0], im_size[:, 1]
        P_norm = P.clone()
        # normalize Y
        P_norm[:, 0, :] = cls.unnormalize_axis(P[:, 0, :], w.unsqueeze(1).expand_as(P[:, 0, :]))
        # normalize X
        P_norm[:, 1, :] = cls.unnormalize_axis(P[:, 1, :], h.unsqueeze(1).expand_as(P[:, 1, :]))
        return P_norm

    @classmethod
    def PointsToUnitCoords(cls, P, im_size):
        h, w = im_size[:, 0], im_size[:, 1]
        P_norm = P.clone()
        # normalize Y
        P_norm[:, 0, :] = cls.normalize_axis(P[:, 0, :], w.unsqueeze(1).expand_as(P[:, 0, :]))
        # normalize X
        P_norm[:, 1, :] = cls.normalize_axis(P[:, 1, :], h.unsqueeze(1).expand_as(P[:, 1, :]))
        return P_norm

    # corr to matches copy from ANCNet
    @classmethod
    def corr_to_matches(cls,
            corr4d,
            delta4d=None,
            k_size=1,
            do_softmax=False,
            scale="centered",
            return_indices=False,
            invert_matching_direction=False,
    ):
        """
        corr_to_matches() interprente the corr4d into correspondences defined in the centred/postive unit coordinate system
        Arguements:
            corr4d:
        Returns:
            xA, yA, xB, yB, B x 256 ; A are regular unit coordinate form the source feature map,
                            B are best matches of A in image B
            scores: B x 256; number of correlations
        """
        to_cuda = lambda x: x.cuda() if corr4d.is_cuda else x
        batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

        if scale == "centered":
            XA, YA = np.meshgrid(
                np.linspace(-1, 1, fs2 * k_size), np.linspace(-1, 1, fs1 * k_size)
            )
            XB, YB = np.meshgrid(
                np.linspace(-1, 1, fs4 * k_size), np.linspace(-1, 1, fs3 * k_size)
            )
        elif scale == "positive":
            XA, YA = np.meshgrid(
                np.linspace(0, 1, fs2 * k_size), np.linspace(0, 1, fs1 * k_size)
            )
            XB, YB = np.meshgrid(
                np.linspace(0, 1, fs4 * k_size), np.linspace(0, 1, fs3 * k_size)
            )

        JA, IA = np.meshgrid(range(fs2), range(fs1))
        JB, IB = np.meshgrid(range(fs4), range(fs3))

        XA, YA = (
            Variable(to_cuda(torch.FloatTensor(XA))),
            Variable(to_cuda(torch.FloatTensor(YA))),
        )
        XB, YB = (
            Variable(to_cuda(torch.FloatTensor(XB))),
            Variable(to_cuda(torch.FloatTensor(YB))),
        )

        JA, IA = (
            Variable(to_cuda(torch.LongTensor(JA).view(1, -1))),
            Variable(to_cuda(torch.LongTensor(IA).view(1, -1))),
        )
        JB, IB = (
            Variable(to_cuda(torch.LongTensor(JB).view(1, -1))),
            Variable(to_cuda(torch.LongTensor(IB).view(1, -1))),
        )

        if invert_matching_direction:
            nc_A_Bvec = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

            if do_softmax:
                nc_A_Bvec = torch.nn.functional.softmax(nc_A_Bvec, dim=3)

            match_A_vals, idx_A_Bvec = torch.max(nc_A_Bvec, dim=3)
            score = match_A_vals.view(batch_size, -1)

            iB = IB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
            jB = JB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
            iA = IA.expand_as(iB)
            jA = JA.expand_as(jB)

        else:
            # actually runs
            nc_B_Avec = corr4d.view(
                batch_size, fs1 * fs2, fs3, fs4
            )  # [batch_idx,k_A,i_B,j_B]
            if do_softmax:
                nc_B_Avec = torch.nn.functional.softmax(nc_B_Avec, dim=1)

            match_B_vals, idx_B_Avec = torch.max(nc_B_Avec, dim=1)
            score = match_B_vals.view(batch_size, -1)

            iA = IA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
            jA = JA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
            iB = IB.expand_as(iA)
            jB = JB.expand_as(jA)

        if delta4d is not None:  # relocalization
            delta_iA, delta_jA, delta_iB, delta_jB = delta4d

            diA = delta_iA.squeeze(0).squeeze(0)[
                iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)
            ]
            djA = delta_jA.squeeze(0).squeeze(0)[
                iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)
            ]
            diB = delta_iB.squeeze(0).squeeze(0)[
                iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)
            ]
            djB = delta_jB.squeeze(0).squeeze(0)[
                iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)
            ]

            iA = iA * k_size + diA.expand_as(iA)
            jA = jA * k_size + djA.expand_as(jA)
            iB = iB * k_size + diB.expand_as(iB)
            jB = jB * k_size + djB.expand_as(jB)

        xA = XA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
        yA = YA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
        xB = XB[iB.view(-1), jB.view(-1)].view(batch_size, -1)
        yB = YB[iB.view(-1), jB.view(-1)].view(batch_size, -1)

        if return_indices:
            return (xA, yA, xB, yB, score, iA, jA, iB, jB)
        else:
            return (xA, yA, xB, yB, score)

    @classmethod
    def bilinearInterpPointTnf(cls, matches, target_points_norm):
        """
        bilinearInterpPointTnf()
        Argument:
            matches tuple (xA, yA, xB, yB), xA, yA, xB, yB Bx 256
            target_points_norm [tensor] B x 2 x N
        """
        xA, yA, xB, yB = matches  #

        feature_size = int(np.sqrt(xB.shape[-1]))

        b, _, N = target_points_norm.size()

        X_ = xB.view(-1)  # B*256
        Y_ = yB.view(-1)  # B*256

        grid = torch.FloatTensor(np.linspace(-1, 1, feature_size)).unsqueeze(0).unsqueeze(2)
        # grid is 1 x 16 x 1
        if xB.is_cuda:
            grid = grid.cuda()
        if isinstance(xB, Variable):
            grid = Variable(grid)

        x_minus = (
                torch.sum(
                    ((target_points_norm[:, 0, :] - grid) > 0).long(), dim=1, keepdim=True
                )
                - 1
        )
        x_minus[x_minus < 0] = 0  # fix edge case
        x_plus = x_minus + 1
        x_plus[x_plus > (feature_size - 1)] = feature_size - 1

        y_minus = (
                torch.sum(
                    ((target_points_norm[:, 1, :] - grid) > 0).long(), dim=1, keepdim=True
                )
                - 1
        )
        y_minus[y_minus < 0] = 0  # fix edge case
        y_plus = y_minus + 1
        y_plus[y_plus > (feature_size - 1)] = feature_size - 1

        toidx = lambda x, y, L: y * L + x

        m_m_idx = toidx(x_minus, y_minus, feature_size)
        p_p_idx = toidx(x_plus, y_plus, feature_size)
        p_m_idx = toidx(x_plus, y_minus, feature_size)
        m_p_idx = toidx(x_minus, y_plus, feature_size)

        # print('m_m_idx',m_m_idx)
        # print('p_p_idx',p_p_idx)
        # print('p_m_idx',p_m_idx)
        # print('m_p_idx',m_p_idx)
        topoint = lambda idx, X, Y: torch.cat(
            (
                X[idx.view(-1)].view(b, 1, N).contiguous(),
                Y[idx.view(-1)].view(b, 1, N).contiguous(),
            ),
            dim=1,
        )

        P_m_m = topoint(m_m_idx, X_, Y_)
        P_p_p = topoint(p_p_idx, X_, Y_)
        P_p_m = topoint(p_m_idx, X_, Y_)
        P_m_p = topoint(m_p_idx, X_, Y_)

        multrows = lambda x: x[:, 0, :] * x[:, 1, :]

        f_p_p = multrows(torch.abs(target_points_norm - P_m_m))
        f_m_m = multrows(torch.abs(target_points_norm - P_p_p))
        f_m_p = multrows(torch.abs(target_points_norm - P_p_m))
        f_p_m = multrows(torch.abs(target_points_norm - P_m_p))

        Q_m_m = topoint(m_m_idx, xA.view(-1), yA.view(-1))
        Q_p_p = topoint(p_p_idx, xA.view(-1), yA.view(-1))
        Q_p_m = topoint(p_m_idx, xA.view(-1), yA.view(-1))
        Q_m_p = topoint(m_p_idx, xA.view(-1), yA.view(-1))

        warped_points_norm = (
                                     Q_m_m * f_m_m + Q_p_p * f_p_p + Q_m_p * f_m_p + Q_p_m * f_p_m
                             ) / (f_p_p + f_m_m + f_m_p + f_p_m)
        return warped_points_norm

# 20200831 from hpf
import torch.nn as nn
import torch.nn.functional as F

class PointTPS(nn.Module):
    def __init__(self, kps_a, kps_b, relaxation):
        super(PointTPS, self).__init__()
        l2 = PointTPS.__build_l2_mat(kps_a, kps_a)

        l2[l2 == 0] = 1
        K = l2 * torch.log(l2) * (1/2.0)
        K = K + relaxation * torch.diag(torch.ones(len(kps_a)))
        P = torch.cat([torch.ones(len(kps_a), 1), kps_a], dim=1)

        L_up = torch.cat([K, P], dim=1)
        L_down = torch.cat([P.t(), torch.zeros((3, 3))], dim=1)
        L = torch.cat([L_up, L_down], dim=0)

        v = kps_b - kps_a
        v = torch.cat([v, torch.zeros(3, 2)], dim=0)
        W = L.inverse() @ v

        self.register_buffer('kps_a', kps_a)
        self.register_buffer('W', W)

    def forward(self, pts):
        l2 = PointTPS.__build_l2_mat(self.kps_a, pts)

        l2[(l2 == 0).detach()] = 1
        v = l2 * torch.log(l2) * (1/2.0)

        o = pts[:, 0] - pts[:, 0] + 1

        x = torch.stack([o, pts[:, 0], pts[:, 1]], dim=0)
        x = torch.cat([v, x], dim=0)

        z_x = x.t() @ (self.W[:, 0])
        z_y = x.t() @ (self.W[:, 1])

        target_x = pts[:, 0] + z_x
        target_y = pts[:, 1] + z_y

        return torch.stack([target_x, target_y], dim=1)

    @staticmethod
    def __build_l2_mat(pts_a, pts_b):
        n_a = len(pts_a)
        n_b = len(pts_b)

        m1 = torch.stack([pts_a] * n_b)
        m2 = torch.stack([pts_b] * n_a)

        l2 = ((m1.transpose(0, 1) - m2) ** 2).sum(2)

        return l2


class ImageTPS(PointTPS):
    def __init__(self, kps_a, kps_b, in_size, out_size, relaxation=0):
        super().__init__(kps_b, kps_a, relaxation)
        ws = torch.linspace(0, out_size[0], steps=out_size[0])
        hs = torch.linspace(0, out_size[1], steps=out_size[1])

        self.register_buffer('hs', hs)
        self.register_buffer('ws', ws)
        self.register_buffer('in_size', torch.FloatTensor([in_size]))
        self.out_size = out_size

    def forward(self, img):
        img_h = self.out_size[1]
        img_w = self.out_size[0]
        hs = torch.stack([self.hs] * img_w, dim=1).view(-1)
        ws = torch.stack([self.ws] * img_h, dim=0).view(-1)

        pts = torch.stack([ws, hs], dim=1)
        target_pts = super().forward(pts)
        target_pts = (target_pts - (self.in_size / 2)) / (self.in_size / 2)

        warped_img = F.grid_sample(img.unsqueeze(0), target_pts.view(1, img_h, img_w, 2), mode='bilinear')
        warped_img = warped_img.squeeze(0)

        return warped_img