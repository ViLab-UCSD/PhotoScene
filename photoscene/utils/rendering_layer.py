import torch as th
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
# import os
# import sys
# lpips_dir = os.path.join(os.getenv('REPO_DIR'), 'third_party', 'PerceptualSimilarity')
# sys.path.insert(1, lpips_dir)
import lpips.pretrained_networks as pn
import lpips as util
from collections import namedtuple
from torchvision import models as tv


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_H=64):  # assumes scale factor is same for H and W
    in_H = in_tens.shape[2]
    scale_factor = 1. * out_H / in_H
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', th.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('scale', th.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, lpips=True):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.scaling_layer = ScalingLayer()
        self.in1_feats = None  # pre_cache the features of target image
        self.mask_cache = None  # pre_cache the resized mask
        self.mask_sum = 0.0  # pre_cache the resized mask

        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, mask, retPerLayer=False, pre_cache=False):
        in0_input = self.scaling_layer(in0)
        outs0 = self.net.forward(in0_input)
        feats0, feats1, diffs = {}, {}, {}
        self.spatial = mask is not None

        if self.in1_feats is None or not pre_cache:
            in1_input = self.scaling_layer(in1)
            with th.no_grad():
                outs1 = self.net.forward(in1_input)
                for kk in range(self.L):
                    feats1[kk] = util.normalize_tensor(outs1[kk])
            self.in1_feats = feats1

        for kk in range(self.L):
            feats0[kk] = util.normalize_tensor(outs0[kk])
            diffs[kk] = (feats0[kk] - self.in1_feats[kk])**2

        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk].model(diffs[kk]), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if self.spatial:
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_H=in0.shape[2]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = res[0]
        for ll in range(1, self.L):
            val += res[ll]
        if mask is not None:
            if self.mask_cache is None or not pre_cache:
                mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
                mask = F.interpolate(mask, size=val.shape[2:], mode="bilinear", align_corners=True)
                self.mask_cache = mask / th.sum(mask)
            if val.shape[0] == 1:
                val = th.sum(val * self.mask_cache)
            else:
                val = th.sum(val * self.mask_cache, dim=(1, 2, 3))
        if retPerLayer:
            return val, res
        else:
            return val


class vgg16_style(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16_style, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(1):  # conv1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 6):  # conv2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 11):  # conv3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 18):  # conv4_1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 25):  # conv5_1
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_conv1_1 = h
        h = self.slice2(h)
        h_conv2_1 = h
        h = self.slice3(h)
        h_conv3_1 = h
        h = self.slice4(h)
        h_conv4_1 = h
        h = self.slice5(h)
        h_conv5_1 = h
        vgg_outputs = namedtuple("VggOutputs", ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
        out = vgg_outputs(h_conv1_1, h_conv2_1, h_conv3_1, h_conv4_1, h_conv5_1)

        return out


class StyleLoss(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, lpips=True):
        super(StyleLoss, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.scaling_layer = ScalingLayer()
        self.in1_feats = None  # pre_cache the features of target image
        self.mask_cache = None  # pre_cache the resized mask
        self.mask_sum = 0.0  # pre_cache the resized mask

        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = vgg16_style
            self.chns = [64, 128, 256, 512, 512]

        self.L = len(self.chns)

        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

    def gram_matrix_masked(self, input, mask):
        # https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
        a, b, c, d = input.size()
        mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
        mask = F.interpolate(mask.float(), size=input.size()[2:], mode="bilinear", align_corners=True)
        mask = mask > 0.9
        features = input.view(a, b, c * d)[:, :, mask.view(-1)]
        G = th.matmul(features, features.permute(0, 2, 1))  # a x b x b
        if th.sum(mask) > 0:
            return G.div(b * th.sum(mask))
        else:
            return th.zeros_like(G)

    def forward(self, in0, in1, mask):
        in0_input = self.scaling_layer(in0)
        in1_input = self.scaling_layer(in1)
        val = 0
        # with th.no_grad():
        outs0 = self.net.forward(in0_input)
        outs1 = self.net.forward(in1_input)
        for kk in range(self.L):
            G0 = self.gram_matrix_masked(outs0[kk], mask)
            G1 = self.gram_matrix_masked(outs1[kk], mask)
            diff = th.mean((G0 - G1)**2, dim=(1, 2))
            val += diff
        return val


class MicrofacetUV:
    def __init__(self, res, inputData, imWidth=160, imHeight=120, fov=57, F0=0.05,
                cameraPos=[0, 0, 0], envWidth=16, envHeight=8, onlyMean=False,
                useVgg=False, useStyle=False, isHdr=False, device='cuda'):
        normalInit = inputData['normal']
        envmapInit = inputData['envmap']
        uvCoord     = inputData['uvInput']
        snMask      = inputData['lossMask']
        lossWeightMap = inputData['lossWeightMap']

        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.fov = fov / 180.0 * np.pi
        self.F0 = F0
        self.cameraPos = th.from_numpy(np.array(cameraPos, dtype=np.float32))
        self.device = device
        self.bbox = [0, 0, 1, 1]
        self.snMask = snMask > 0.9
        self.uvCoord = uvCoord  # 1 x H x W x 2
        self.res = res  # resolution of texture
        self.eps = 1e-6

        self.initGeometry()
        self.Rot = self.getRot(th.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).repeat(imHeight, imWidth, 1).to(self.device),
                                normalInit.squeeze(0).permute(1, 2, 0).to(self.device))
        self.initEnv()
        self.up = th.Tensor([0, 1, 0]).to(self.device)

        self.v = self.v.to(device)
        self.pos = self.pos.to(device)
        self.up = self.up.to(device)
        self.ls = self.ls.to(device)
        self.envWeight = self.envWeight.to(device)
        self.lossWeightMap = lossWeightMap  # 1 x 1 x H x W
        self.onlyMean = onlyMean  # For pixel statistic L1 loss
        self.isHdr = isHdr

        if useVgg:
            self.pnet = PNetLin(use_dropout=False, spatial=True, lpips=False).to(device)
        if useStyle:
            self.styleLoss = StyleLoss(use_dropout=False, spatial=True, lpips=False).to(device)

        self.envmapInit = envmapInit

    def getRot(self, a, b):  # get rotation matrix from vec a to b
        # a: size H x W x 3, b: size H x W x 3
        # output: H x W x 3 x 3
        H, W, _ = a.shape
        zeros = th.zeros((H, W)).to(self.device)
        a = a / th.linalg.norm(a, dim=2).unsqueeze(2)
        b = b / th.linalg.norm(b, dim=2).unsqueeze(2)
        nu = th.cross(a, b, dim=2)
        nu_skew = th.stack([zeros, -nu[:, :, 2], nu[:, :, 1],
                            nu[:, :, 2], zeros, -nu[:, :, 0],
                            -nu[:, :, 1], nu[:, :, 0], zeros], dim=2).view(H, W, 3, 3)
        c = th.sum(a * b, dim=2).view(H, W, 1, 1).repeat(1, 1, 3, 3)
        R = th.eye(3).view(1, 1, 3, 3).repeat(H, W, 1, 1).to(self.device) + nu_skew + \
            th.matmul(nu_skew, nu_skew) / (1 + c)
        return R  # H x W x 3 x 3

    def initGeometry(self):
        self.xRange = 1 * np.tan(self.fov / 2)
        xStart = 2 * self.xRange * self.bbox[0] - self.xRange
        xEnd   = 2 * self.xRange * (self.bbox[0] + self.bbox[2]) - self.xRange
        self.yRange = float(self.imHeight) / float(self.imWidth) * self.xRange
        yStart = 2 * self.yRange * self.bbox[1] - self.yRange
        yEnd   = 2 * self.yRange * (self.bbox[1] + self.bbox[3]) - self.yRange
        x, y = np.meshgrid(np.linspace(xStart, xEnd, self.imWidth),
                np.linspace(yStart, yEnd, self.imHeight))
        y = np.flip(y, axis=0)
        z = -np.ones( (self.imHeight, self.imWidth), dtype=np.float32)
        self.pos = th.from_numpy(np.stack([x, y, z], 2).astype(np.float32))
        self.pos_norm = self.pos.norm(2.0, 2, keepdim=True)
        self.N = 1
        self.v = self.normalize((self.cameraPos - self.pos).permute(2, 0, 1).unsqueeze(0).expand(self.N, -1, -1, -1))

    def initEnv(self):
        Az = ( (np.arange(self.envWidth) + 0.5) / self.envWidth - 0.5) * 2 * np.pi
        El = ( (np.arange(self.envHeight) + 0.5) / self.envHeight) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az.reshape(-1, 1)
        El = El.reshape(-1, 1)
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis=1)

        envWeight = np.sin(El) * np.pi * np.pi / self.envWidth / self.envHeight

        self.ls = th.from_numpy(ls.astype(np.float32))  # envWidth * envHeight, 3
        self.envWeight = th.from_numpy(envWeight.astype(np.float32))
        self.envWeight = self.envWeight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def GGX(self, cos_h, alpha):
        c2 = cos_h ** 2
        a2 = alpha ** 2
        den = c2 * a2 + (1 - c2)
        return a2 / (np.pi * den**2 + self.eps)

    def Fresnel(self, cos, f0):
        sphg = th.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos)
        return f0 + (1.0 - f0) * sphg

    def G(self, n_dot_v, n_dot_l, rough):
        def _G1(cos, k):
            return cos / (cos * (1.0 - k) + k)

        k = ((rough + 1) ** 2) / 8.0
        return _G1(n_dot_v, k) * _G1(n_dot_l, k)

    def normalize(self, vec):
        assert (vec.size(0) == self.N)
        assert (vec.size(1) == 3)
        assert (vec.size(2) == self.imHeight)
        assert (vec.size(3) == self.imWidth)

        vec = vec / (vec.norm(2.0, 1, keepdim=True))
        return vec

    def getDir(self):
        vec = (self.cameraPos - self.pos).permute(2, 0, 1).unsqueeze(0).expand(self.N, -1, -1, -1)
        return self.normalize(vec), (vec**2).sum(1, keepdim=True).expand(-1, 3, -1, -1)

    def AdotB(self, a, b):
        ab = th.clamp(th.sum(a * b, dim=2), 0, 1).unsqueeze(2)
        return ab

    def fromUV(self, valMap, uvDict=None):
        if uvDict is None:
            uvDict = {'scale': th.tensor([1.0, 1.0]).to(self.device), 'rot': th.tensor([0.0]).to(self.device),
                        'trans': th.tensor([0.0, 0.0]).to(self.device)}
        else:
            assert (uvDict['rot'] is not None and uvDict['trans'] is not None and
                    uvDict['scaleBase'] is not None and uvDict['logscale'] is not None)
            uvDict['scale'] = th.pow(uvDict['scaleBase'], uvDict['logscale'])

        grid = self.uvCoord
        # Apply UV transformation #
        uvScale = uvDict['scale']
        grid = grid * uvScale
        c, s = th.cos(uvDict['rot']), th.sin(uvDict['rot'])
        uvRot = th.stack([th.cat([c, -s]), th.cat([s, c])], dim=0)
        grid = th.matmul(uvRot, grid.unsqueeze(-1)).squeeze(-1)
        uvTrans = uvDict['trans']
        grid = grid + uvTrans
        # Convert to -1, 1 #
        grid = grid - th.floor(grid)
        grid = (grid * 2 - 1).clamp(-1 + self.eps, 1 - self.eps)
        return th.nn.functional.grid_sample(valMap, grid.expand(valMap.shape[0], -1, -1, -1), mode='bilinear',
            padding_mode='border', align_corners=False)

    def getBinaryMask(self,):
        return self.snMask

    def applyMaskLoss(self, aMap, bMap):
        _, _, H, W = self.snMask.shape
        mask = self.snMask.view(-1)
        _, aC = aMap.shape[0], aMap.shape[1]
        _, bC = bMap.shape[0], bMap.shape[1]
        _, weightC = self.lossWeightMap.shape[0], self.lossWeightMap.shape[1]
        aMap = aMap.view(-1, aC, H * W)[:, :, mask]
        bMap = bMap.view(-1, bC, H * W)[:, :, mask]
        return ( (aMap - bMap) * self.lossWeightMap.view(-1, weightC, H * W)[:, :, mask]).abs().mean(dim=(1, 2))

    def applyMaskStatLoss(self, orMap, snMap, w1=1, w2=1):
        _, _, H, W = self.snMask.shape
        orB, orC = orMap.shape[0], orMap.shape[1]
        snB, snC = snMap.shape[0], snMap.shape[1]
        _, weightC = self.lossWeightMap.shape[0], self.lossWeightMap.shape[1]
        snMask = self.snMask.view(-1)
        orMap = orMap.view(orB, orC, H * W)[:, :, snMask]
        snMap = snMap.view(snB, snC, H * W)[:, :, snMask]
        loss1 = (th.mean(orMap * self.lossWeightMap.view(-1, weightC, H * W)[:, :, snMask], dim=2)
            - th.mean(snMap * self.lossWeightMap.view(-1, weightC, H * W)[:, :, snMask], dim=2)).abs().mean(dim=1)
        if self.onlyMean:
            return loss1
        else:
            loss2 = (th.var(orMap * self.lossWeightMap.view(-1, weightC, H * W)[:, :, snMask], dim=2)
                - th.var(snMap * self.lossWeightMap.view(-1, weightC, H * W)[:, :, snMask], dim=2)).abs().mean(dim=1)
            return w1 * loss1 + w2 * loss2

    def applyMaskVggLoss(self, orMap, snMap):
        return self.pnet.forward(orMap.float() * self.snMask.float(), snMap.float() * self.snMask.float(),
            self.lossWeightMap.squeeze())

    def applyMaskStyleLoss(self, orMap, snMap):
        return self.styleLoss.forward(orMap.float() * self.snMask.float(), snMap.float() * self.snMask.float(),
            self.lossWeightMap.squeeze())

    def eval(self, albedoMap, normalMap, roughMap, uvDict=None, returnBRDF=False, eps=1e-10):
        # albedo, normal, rough: B x c x res x res
        # -> albedo, normal, rough: B x c x H x W
        albedo = self.fromUV(albedoMap, uvDict=uvDict)
        rough  = self.fromUV(roughMap, uvDict=uvDict)
        normalInit = self.fromUV(normalMap, uvDict=uvDict)

        ldirections = self.ls.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # 1, envWidth * envHeight, 3, 1, 1
        normalPred = th.matmul(
            self.Rot, normalInit.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2).to(self.device)
        # print(normalPred.shape, self.up.shape) # 1 3 256 256, 3
        camyProj = th.einsum('b,abcd->acd', (self.up, normalPred)).unsqueeze(1).expand_as(normalPred) * normalPred
        camy = th.nn.functional.normalize(
            self.up.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(camyProj) - camyProj, dim=1)
        camx = -th.nn.functional.normalize(th.cross(camy, normalPred, dim=1), p=1, dim=1)
        ll = ldirections[:, :, 0:1, :, :] * camx.unsqueeze(1) \
                + ldirections[:, :, 1:2, :, :] * camy.unsqueeze(1) \
                + ldirections[:, :, 2:3, :, :] * normalPred.unsqueeze(1)  # 1, envWidth * envHeight, 3, 1, 1

        if returnBRDF:
            return albedo ** (1 / 2.2), rough, normalInit.clamp(-1, 1) * 0.5 + 0.5, normalPred.clamp(-1, 1) * 0.5 + 0.5

        normal = normalPred.unsqueeze(1)

        h = (self.v + ll) / 2
        h = h / th.sqrt(th.clamp(th.sum(h * h, dim=2), min=1e-6).unsqueeze(2))

        n_dot_v = self.AdotB(normal, self.v)
        n_dot_l = self.AdotB(normal, ll)
        n_dot_h = self.AdotB(normal, h)
        v_dot_h = self.AdotB(self.v, h)

        D = self.GGX(n_dot_h, rough.unsqueeze(1)**2)
        F = self.Fresnel(v_dot_h, self.F0)
        G = self.G(n_dot_v, n_dot_l, rough.unsqueeze(1))

        # lambert brdf
        f1 = albedo.unsqueeze(1) / np.pi

        # cook-torrence brdf
        f2 = D * F * G / (4 * n_dot_v * n_dot_l + self.eps)

        # brdf
        kd, ks = 1, 1
        f = kd * f1 + ks * f2

        # rendering
        envmap = self.envmapInit  # [1, 3, 119, 160, 8, 16]
        envR, envC = envmap.size(2), envmap.size(3)
        envmap = envmap.view([1, 3, envR, envC, self.envWidth * self.envHeight])
        envmap = envmap.permute([0, 4, 1, 2, 3])  # [1, self.envWidth * self.envHeight, 3, envR, envC]
        envmap = th.nn.functional.interpolate(envmap.squeeze(0), size=(self.imHeight, self.imWidth))
        # [self.envWidth * self.envHeight, 3, imHeight, imWidth]
        envmap = envmap[:, :, int(self.imHeight * self.bbox[1]):int(self.imHeight * (self.bbox[1] + self.bbox[3])),
                                int(self.imWidth * self.bbox[0]):int(self.imWidth * (self.bbox[0] + self.bbox[2]))]
        self.envmap = envmap.unsqueeze(0).to(self.device)

        img = f * n_dot_l * self.envmap * self.envWeight.expand(
            [1, self.envWidth * self.envHeight, 3, self.imHeight, self.imWidth])
        img = th.sum(img, dim=1)
        if self.isHdr:
            return img.clamp(min=eps).float()
        else:
            return img.clamp(eps, 1) ** (1 / 2.2)
