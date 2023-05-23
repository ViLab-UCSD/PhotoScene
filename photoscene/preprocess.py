import os
import os.path as osp
import pickle
import argparse
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
import math
import torch as th
from scipy import ndimage

from utils.io import load_cfg, loadBinary, readModelStrIdDict, readViewDict, readViewList, readNormal, \
    loadInvRenderData, modifyMaterialName
from utils.xml import createXmlFromInfoDict, saveNewModelIdFileAndXmlFile, saveNewWhiteXML, getBsdfIdDict, \
    saveNewBaselineXML, genSingleLightAreaXMLs
from utils.render import renderPass

import scipy.io
import random


def saveViewList(sampledIdList, sampledViewFile):
    if not osp.exists(osp.dirname(sampledViewFile)):
        os.makedirs(osp.dirname(sampledViewFile))
    with open(sampledViewFile, 'w') as f:
        f.writelines('%d ' % vId for vId in sampledIdList)
    print('Sampled views are saved at %s' % sampledViewFile)


def saveViewDict(selectedViewFile, matViewDict):
    if not osp.exists(osp.dirname(selectedViewFile)):
        os.makedirs(osp.dirname(selectedViewFile))
    with open(selectedViewFile, 'w') as f:
        for mat in sorted(matViewDict.keys()):
            f.writelines('%s %d\n' % (mat, matViewDict[mat]))
    # print('Selected views are saved at %s' % selectedViewFile)


def loadSceneRenderData(cfg, vId, refH, refW):
    def depthToNormal(depth, fov=57, eps=1e-6):  # 1 x 1 x H x W
        z = depth.squeeze()  # H x W
        validMask = (z > 0)
        H, W = z.shape
        xScale = np.tan((fov / 180.0 * np.pi) / 2.0)
        yScale = xScale / W * H
        zx = z * xScale / (W / 2.0)
        zy = z * yScale / (H / 2.0)
        dzdx = (z[:, 2:] - z[:, :W - 2]) / 2.0 / (zx[:, 1:W - 1] + eps)  # H x (W-2)
        dzdy = (z[2:, :] - z[:H - 2, :]) / 2.0 / (zy[1:H - 1, :] + eps)  # (H-2) x W
        direction = th.stack([dzdx[1:H - 1, :], -dzdy[:, 1:W - 1], th.ones((H - 2, W - 2))], dim=0)
        normal = direction / th.sqrt(th.sum(direction ** 2.0, dim=0)).unsqueeze(0)  # 3 x H-2 x W-2
        pad = th.nn.ZeroPad2d((1, 1, 1, 1))
        normal = pad(normal)
        return normal.unsqueeze(0), validMask.unsqueeze(0).unsqueeze(1)  # 1 x 3 x H x W, 1 x 1 x H x W
    # load openroom uv coords
    uvPath = cfg.uvPathById % vId
    uvMap = loadBinary(uvPath, channels=2, if_resize=True, imWidth=refW, imHeight=refH)  # 1 x refH x refW x 2
    uvMapTensor = th.from_numpy(uvMap)

    # load openroom normal
    # orNormalImg = cfg.normalPathById % vId
    # orNormal = readNormal(orNormalImg, refW, refH)
    depthPath = cfg.depthPathById % vId
    depthMap = loadBinary(depthPath, channels=1, if_resize=True, imWidth=refW * 2, imHeight=refH * 2)
    # 1 x refH*2 x refW*2
    depthMap = th.from_numpy(depthMap).unsqueeze(1)  # 1 x 1 x refH*2 x refW*2
    orNormalFromDepth, validMask = depthToNormal(depthMap)  # 1 x 3 x refH*2 x refW*2
    orNormalFromDepth = orNormalFromDepth * validMask.float()
    orNormalFromDepth = th.nn.functional.interpolate(orNormalFromDepth, scale_factor=(0.5, 0.5),
                                                        recompute_scale_factor=True)

    orData = {  'uv'    : uvMapTensor,
                'normal': orNormalFromDepth}

    return orData


def selectViewAndSaveMask(cfg, candidateCamIdList, bsdfIdDict, imWidth, imHeight, labelMapDict, normalTH=0.9,
                            isOverwrite=False):
    def readMatPartMaskBatch(filePath, bsdfIdDict, if_resize=False, imWidth=None, imHeight=None, pixNumTH=100):
        matMap = loadBinary(filePath, channels=3, dtype=np.int32, if_resize=if_resize, imWidth=imWidth,
                            imHeight=imHeight).squeeze()
        cadIdMap = matMap[:, :, 0]
        matIdMap = matMap[:, :, 1]
        insIdMap = matMap[:, :, 2]
        partMaskDict = {}
        insMaskDict = {}
        for mat in bsdfIdDict.keys():
            partIdDict = bsdfIdDict[mat]
            cadId, matId = partIdDict['cadId'], partIdDict['matId']
            cadMask = cadIdMap == cadId
            matMask = matIdMap == matId
            partMask = cadMask * matMask
            if np.sum(partMask.astype(np.int32)) < pixNumTH:
                continue
            partMaskDict[mat] = partMask
            if 'scene' in mat:  # separate each plane of wall to diff. segs
                insMaskOpt = insIdMap * cadMask * matMask
                camIdOpt = filePath.split('_')[-1].split('.')[0]
                normalFile = os.path.join(cfg.normalRenderDir, cfg.normalPathById % int(camIdOpt))
                normal = readNormal(normalFile, imWidth, imHeight)
                orIns = th.from_numpy(insMaskOpt).unsqueeze(0).unsqueeze(1)
                segList = segFromNormal(normal, orIns, threshold=0.95)
                insMaskOpt = th.zeros_like(orIns).view(-1)
                for idx, seg in enumerate(segList):
                    insMaskOpt[seg.view(-1)] = idx + 1
                insMaskOpt = insMaskOpt.view(1, 1, imHeight, imWidth)
            elif '03636649' in mat:  # lamp, combine light source with holder
                insMaskOpt = th.from_numpy(cadMask).unsqueeze(0).unsqueeze(1)
            elif '02818832' in mat:  # bed, decompose it to bed and pillows as different instances
                # check if mat part it's small part of the instance, e.g.pillow,
                # if yes, further segment, if no, keep original
                insPart = th.from_numpy(insIdMap * partMask).unsqueeze(0).unsqueeze(1)
                insCad = th.from_numpy(insIdMap * cadMask).unsqueeze(0).unsqueeze(1)
                if th.sum(insPart) < 0.5 * th.sum(insCad):
                    insMaskOpt = insPart
                else:
                    insMaskOpt = insCad
            else:
                insMaskOpt = th.from_numpy(insIdMap * cadMask).unsqueeze(0).unsqueeze(1)
            insMaskDict[mat] = insMaskOpt
        return partMaskDict, insMaskDict

    def readSegLabel(labelFile, if_resize=False, imWidth=None, imHeight=None):
        label = np.array(Image.open(labelFile))
        label = th.from_numpy(label).unsqueeze(0).unsqueeze(1)
        if if_resize:
            label = th.nn.functional.interpolate(label, size=(imHeight, imWidth), mode='nearest')
        return label.type(th.int)

    def readGtMask(maskFile, imWidth=None, imHeight=None, readAlpha=False):
        if not osp.exists(maskFile):
            print('%s not exist!' % maskFile)
            assert (False)
        photoMask = cv2.imread(maskFile, cv2.IMREAD_UNCHANGED)
        if readAlpha:
            assert (photoMask.shape[2] == 4)
            photoMask = photoMask[:, :, 3]  # read alpha channel
        else:
            if len(photoMask.shape) == 3:
                photoMask = photoMask[:, :, 0]
            else:
                assert (len(photoMask.shape) == 2)
                photoMask = photoMask
        snIns = th.tensor(photoMask).unsqueeze(0).unsqueeze(1)
        snIns = th.nn.functional.interpolate(snIns, size=(imHeight, imWidth), mode='nearest') > 0
        return snIns.type(th.int)

    def getOrInsList(mat, insMaskDict):
        insMaskOpt = insMaskDict[mat]
        orIns = insMaskOpt
        orInsList = []
        for orId in list(th.unique(orIns)):
            if orId == 0:
                continue
            m = orIns == orId
            orInsList.append(m)
        return orInsList

    def getSnInsMap(mat, matPartMask, labelMapDict, semLabel, insLabel):
        # Use OR matPartMask to search corresponding SN mask among candidates from semantic masks
        # Find the candidate with best MIOU, then multipy with insLabel to output ins. label for correct class
        modelID = mat.split('_')[0]
        if 'scene' in modelID:
            modelID = mat.split('_')[2]
        elif 'ceiling' in modelID:
            modelID = 'ceiling_lamp'
        if modelID in labelMapDict.keys():
            snIDList = labelMapDict[modelID]  # map openroom id to scannet id
        else:
            snIDList = th.unique(semLabel).tolist()
        semBest = th.zeros_like(semLabel)
        # Find the semantic mask which has best mIoU with matPartMask ###
        bestMIOU = 0
        for snID in snIDList:
            semMask = semLabel == snID
            # use mIoU
            matPartMaskSoft = toSoftMask(matPartMask)
            intersection = th.sum(matPartMaskSoft * semMask.float())
            union = th.sum(matPartMaskSoft + semMask.float())
            mIoU = intersection / union
            if mIoU > bestMIOU:
                bestMIOU = mIoU
                semBest = semMask

        if bestMIOU == 0:
            snIns = insLabel * semBest
        else:
            snIns = insLabel * semBest

        return snIns

    def getSnInsList(snIns):
        insList = []
        for snId in list(th.unique(snIns)):
            if snId == 0:
                continue
            m = snIns == snId
            insList.append(m)
        return insList

    def segFromNormal(normal, mask, threshold=0.95):
        def findConnectedMask(mask, nPixThreshold=100):
            num_labels, labels_im = cv2.connectedComponents(mask.byte().squeeze().cpu().numpy() * 255)
            segMasks = []
            for lab in range(num_labels):
                segMask = labels_im == lab
                if np.sum(segMask) > nPixThreshold and (lab != 0):
                    # print('SegMask sum:', np.sum(segMask))
                    segMasks.append(th.tensor(segMask.astype(np.int32)).unsqueeze(0).unsqueeze(1).bool())  # bool
            return segMasks
        # normal: 1 x 3 x H x W; mask: 1 x 1 x H x W
        # output: list of segs with similar normal directions
        mask = mask > 0.5
        normalMasked = normal.view(3, -1)[:, mask.view(-1)]  # 3 x nPix

        # kmeans: find 1~3 plane normals
        kmeans = KMeans(n_clusters=3, random_state=cfg.seed).fit(normalMasked.permute(1, 0).cpu().numpy())
        cluster_centers = th.from_numpy(kmeans.cluster_centers_)
        # print(cluster_centers)

        cond01 = th.sum(cluster_centers[0] * cluster_centers[1]) > threshold
        cond02 = th.sum(cluster_centers[0] * cluster_centers[2]) > threshold
        cond12 = th.sum(cluster_centers[1] * cluster_centers[2]) > threshold
        if cond01 or cond02 or cond12:
            if cond01 and ~cond02 and ~cond12:
                new_c = (cluster_centers[0] + cluster_centers[1]) / 2.0
                cluster_centers = th.stack([new_c, cluster_centers[2]], dim=0)
            elif cond02 and ~cond01 and ~cond12:
                new_c = (cluster_centers[0] + cluster_centers[2]) / 2.0
                cluster_centers = th.stack([new_c, cluster_centers[1]], dim=0)
            elif cond12 and ~cond02 and ~cond01:
                new_c = (cluster_centers[1] + cluster_centers[2]) / 2.0
                cluster_centers = th.stack([new_c, cluster_centers[0]], dim=0)
            else:
                cluster_centers = th.sum(cluster_centers, dim=0).view(1, 3) / 3.0

        segMasks = []
        for cId in range(cluster_centers.size(0)):
            center = cluster_centers[cId].view(1, 3, 1, 1)
            seg = th.sum(normal * center.to(normal.device), dim=1, keepdim=True) > threshold
            planeMask = seg * mask
            planeSegMasks = findConnectedMask(planeMask)
            segMasks += planeSegMasks

        return segMasks

    def mask2bound(mask):
        H, W = mask.squeeze().shape
        x_inds = th.nonzero(th.any(mask.squeeze(), axis=0))  # HxW -> W -> #nonzero
        x_min = ( (x_inds[0] + 0.5) / W - 0.5) * 2.0  # --> [-1, 1]
        x_max = ( (x_inds[-1] + 0.5) / W - 0.5) * 2.0
        y_inds = th.nonzero(th.any(mask.squeeze(), axis=1))  # HxW -> H -> #nonzero
        y_min = ( (y_inds[0] + 0.5) / H - 0.5) * 2.0
        y_max = ( (y_inds[-1] + 0.5) / H - 0.5) * 2.0
        x_c, y_c = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        x_l, y_l = (x_max - x_min), (y_max - y_min)
        return th.tensor([x_c, y_c]), th.tensor([x_l, y_l])  # center, length

    def toSoftMask(mask):  # apply gaussian filter around the mask
        # assume only 1 connected component in mask
        H, W = mask.squeeze().shape
        center, length = mask2bound(mask)
        meshgrids = th.meshgrid(
            [ ( (th.arange(size, dtype=th.float32) + 0.5) / size - 0.5) * 2.0 for size in [H, W]],
            indexing='ij'
        )
        means = [center[1], center[0]]
        sigma = [length[1] / 2.0, length[0] / 2.0]
        kernel = 1
        for mean, std, mgrid in zip(means, sigma, meshgrids):
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                th.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / th.sum(kernel) * th.sum(mask)
        return kernel.view(1, 1, H, W)

    def matchInsList(orInsList, snInsList):
        # Match among Openroom and ScanNet instance lists
        # Match from Openroom large instance
        orInsSelect = []
        snInsSelect = []
        insCnt = min(len(orInsList), len(snInsList))
        if len(orInsList) == 1 and len(snInsList) == 1:
            orInsSelect.append(orInsList[0])
            snInsSelect.append(snInsList[0])
        else:
            orInsListPixNum = {}
            for orInsMask in orInsList:
                pixNum = th.sum(orInsMask.float())
                orInsListPixNum[pixNum] = orInsMask

            cnt = 0
            snInsListCopy = snInsList.copy()
            for pixNum in sorted(orInsListPixNum.keys(), reverse=True):
                orInsMask = orInsListPixNum[pixNum]
                mIoUMax = 0
                selectIdx = -1
                for idx2, snInsMask in enumerate(snInsListCopy):  # snMask
                    orInsMaskSoft = toSoftMask(orInsMask)
                    snInsMaskSoft = toSoftMask(snInsMask)
                    intersection = th.sum(orInsMaskSoft * snInsMaskSoft)
                    union = th.sum(orInsMaskSoft + snInsMaskSoft)
                    mIoU = intersection / union
                    if mIoU > mIoUMax and mIoU > 0:
                        mIoUMax = mIoU
                        snInsMaskOpt = snInsMask
                        selectIdx = idx2
                if mIoUMax > 0:
                    orInsSelect.append(orInsMask)
                    snInsSelect.append(snInsMaskOpt)
                    if selectIdx > -1:
                        del snInsListCopy[selectIdx]
                    cnt += 1
                    if cnt == insCnt:
                        break
        return orInsSelect, snInsSelect

    def findConsensus(invRenderInitStatDict, inTH=0.2):  # {mat: [{vId:Z alb: X, rou:Y } ] }
        selectedViewDict = {}
        num = len(invRenderInitStatDict.keys())
        for mm, mat in enumerate(invRenderInitStatDict.keys()):
            print('[%d/%d] Finding Consensus for material %s ...' % (mm + 1, num, mat))
            matStatList = invRenderInitStatDict[mat]
            optView = -1
            inlierCntList = []
            pixNumList = []
            pixNumOriList = []
            optPixNum = -1
            for matData in matStatList:  # {vId: {alb: X, rou: Y}}
                vId = matData['vId']
                albMean, rouMean = matData['albMean'], matData['roughMean']
                albVar, rouVar = matData['albVar'], matData['roughVar']
                pixNum = matData['pixNum']
                pixNumOri = matData['pixNumOri']
                cnt = 0
                for matData2 in matStatList:
                    vId2 = matData2['vId']
                    if vId2 == vId:
                        continue
                    albMean2, rouMean2 = matData2['albMean'], matData2['roughMean']
                    albVar2, rouVar2 = matData2['albVar'], matData2['roughVar']
                    err1 = (th.norm(albMean - albMean2) + th.norm(rouMean - rouMean2))
                    err2 = (th.norm(albVar - albVar2) + th.norm(rouVar - rouVar2))
                    err = err1 + err2
                    if err < inTH:
                        cnt += 1
                inlierCntList.append(cnt)
                pixNumList.append(pixNum)
                pixNumOriList.append(pixNumOri)

            optVal = 0
            for idx, matData in enumerate(matStatList):
                vId = matData['vId']
                val = (inlierCntList[idx] + 1) * pixNumList[idx]
                if val > optVal:
                    optView = vId
                    optVal = val
                    optPixNum = pixNumOriList[idx]

            if optView > -1 and optPixNum > 0:
                selectedViewDict[mat] = optView
                print('        Selected optimal view ID: %d for material: %s, pixel num is %d' %
                    (optView, mat, optPixNum))
            else:
                print('        Optimal view for material %s not found!' % mat)
        return selectedViewDict

    def transformInp(inp, center1, center2, length1, length2):
        # center1: orCenter; center2: snCenter
        # transform from 1 to 2
        # inp: N x C x H x W
        theta = th.tensor([[1., 0., 0.], [0., 1., 0.]])
        grid = th.nn.functional.affine_grid(theta.view(-1, 2, 3), inp.size(), align_corners=False)
        # N x H x W x 2, u v in [-1, 1]
        grid = (grid - center2.view(1, 1, 1, 2)) / length2.view(1, 1, 1, 2) * length1.view(1, 1, 1, 2) \
                + center1.view(1, 1, 1, 2)
        warpedInp = th.nn.functional.grid_sample(inp, grid, align_corners=False)
        return warpedInp

    def jointMask(orMatPart, orNormal, snNormal, normalTH=0.95):
        normalMask = th.sum(orNormal * snNormal, dim=1) > normalTH
        return orMatPart * normalMask

    selectedViewFile = cfg.selectedViewFile
    selectedViewSmallMaskDictFile = cfg.selectedSmallDictFile
    selectedViewNoMaskDictFile = cfg.selectedNoMaskDictFile
    selectedViewFullDictFile = cfg.selectedFullDictFile
    if isOverwrite or not osp.exists(selectedViewFullDictFile):
        invRenderInitStatDict = {}  # {mat: [vId: {alb: X, rou:Y} ] }
        for mat in bsdfIdDict.keys():
            invRenderInitStatDict[mat] = []
        for camId in candidateCamIdList:
            # Search for best view if inv render consensus and with enough number of pixels
            partMaskFile = cfg.partIdPathById % int(camId)
            matPartMaskDict, insMaskDict = readMatPartMaskBatch(partMaskFile, bsdfIdDict, if_resize=True,
                imWidth=imWidth, imHeight=imHeight)
            # Read ScanNet semantic and instance labels
            # if labelMapDict is not None: # {modelID: ScanNet class labels}
            if cfg.gtMaskDir is None:
                semLabelFile = cfg.semLabById % camId
                semLabel = readSegLabel(semLabelFile, if_resize=True, imWidth=imWidth, imHeight=imHeight)
                insLabelFile = cfg.insLabById % camId
                insLabel = readSegLabel(insLabelFile, if_resize=True, imWidth=imWidth, imHeight=imHeight)
            # Read Inverse Rendering Pred.
            invRenderData = loadInvRenderData(cfg, camId, refH=imHeight, refW=imWidth, device='cpu')
            # Collect Inverse Rendering init. results for each material
            for mat, matPartMask in matPartMaskDict.items():
                # load openroom instance mask list
                orInsList = getOrInsList(mat, insMaskDict)
                # load scannet instance mask list
                matPartMask = th.from_numpy(matPartMask).unsqueeze(0).unsqueeze(1)
                if cfg.gtMaskDir is None:
                    snIns = getSnInsMap(mat, matPartMask, labelMapDict, semLabel, insLabel)
                else:
                    maskFile = cfg.gtMaskByNameId % (mat, camId)
                    snIns = readGtMask(maskFile, imWidth=imWidth, imHeight=imHeight, readAlpha=True)
                snInsList = getSnInsList(snIns)
                if len(snInsList) == 0:
                    print('\n------>>>>' , mat, ': semantic part not found!\n')
                    continue
                # ----> Step 1: Instance level pair matching
                orInsSelect, snInsSelect = matchInsList(orInsList, snInsList)
                # ----> Step 2: Fetch init. material for ScanNet instance mask
                snInsMask = th.zeros_like(snIns)
                for snIns in snInsSelect:
                    snInsMask = snIns * snIns + (~snIns) * snInsMask
                m = snInsMask.view(-1) > 0
                if th.sum(m) > 0:
                    albMean = th.mean(invRenderData['albedo'].view(3, -1)[:, m], dim=1)
                    albVar = th.var(invRenderData['albedo'].view(3, -1)[:, m], dim=1)
                    roughMean = th.mean(invRenderData['rough'].view(-1)[m])
                    roughVar = th.var(invRenderData['rough'].view(-1)[m])

                    def gaussianMask(mask, mean=0, sig=0.5):
                        # assume only 1 connected component in mask
                        H, W = mask.squeeze().shape
                        meshgrids = th.meshgrid(
                            [ ( (th.arange(size, dtype=th.float32) + 0.5) / size - 0.5) * 2.0 for size in [H, W]],
                            indexing='ij'
                        )
                        means = [mean, mean]
                        sigma = [sig, sig]
                        kernel = 1
                        for mean, std, mgrid in zip(means, sigma, meshgrids):
                            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                                th.exp(-((mgrid - mean) / std) ** 2 / 2)
                        kernel = kernel / th.sum(kernel) * th.sum(mask)
                        return kernel.view(1, 1, H, W)
                    # Apply weight on mask #
                    weightMap = gaussianMask(snInsMask)
                    weightMap = weightMap / weightMap.max()
                    pixNum = th.sum(snInsMask.float() * weightMap.to(snInsMask.device))
                    pixNumOri = th.sum(snInsMask.float())
                    # Apply weight on mask #
                    invRenderInitStatDict[mat].append( {'vId': camId, 'nPix': th.sum(m).item(), 'albMean': albMean,
                        'albVar': albVar, 'roughMean': roughMean, 'roughVar': roughVar, 'pixNum': pixNum,
                        'pixNumOri': pixNumOri})

        # ---> Resolve for consensus <---- #
        selectedViewDict = findConsensus(invRenderInitStatDict)

        # ---> after view selection, warp mask and uv, then save new part mask!
        selectedViewSmallMaskDict = {}
        selectedViewNoMaskDict = {}
        if os.path.exists(selectedViewSmallMaskDictFile):
            selectedViewSmallMaskDict = readViewDict(selectedViewSmallMaskDictFile)
        for mat in selectedViewDict.keys():
            vId = selectedViewDict[mat]
            orData = loadSceneRenderData(cfg, vId, refH=imHeight, refW=imWidth)
            orUvMap = orData['uv'].permute(0, 3, 1, 2)
            orNormal = orData['normal']
            invRenderData = loadInvRenderData(cfg, vId, refH=imHeight, refW=imWidth, device='cpu')
            snNormal = invRenderData['normal']
            if cfg.gtMaskDir is None:
                # Read ScanNet semantic and instance labels
                semLabelFile = cfg.semLabById % vId
                semLabel = readSegLabel(semLabelFile, if_resize=True, imWidth=imWidth, imHeight=imHeight)
                insLabelFile = cfg.insLabById % vId
                insLabel = readSegLabel(insLabelFile, if_resize=True, imWidth=imWidth, imHeight=imHeight)
            # # load openroom instance mask list
            matPartMaskDict, insMaskDict = readMatPartMaskBatch(cfg.partIdPathById % int(vId), bsdfIdDict,
                if_resize=True, imWidth=imWidth, imHeight=imHeight)
            orInsList = getOrInsList(mat, insMaskDict)
            # # load scannet instance mask list
            orMatPart = th.from_numpy(matPartMaskDict[mat]).unsqueeze(0).unsqueeze(1)
            if cfg.gtMaskDir is None:
                snIns = getSnInsMap(mat, orMatPart, labelMapDict, semLabel, insLabel)
            else:
                maskFile = cfg.gtMaskByNameId % (mat, vId)
                snIns = readGtMask(maskFile, imWidth=imWidth, imHeight=imHeight, readAlpha=True)
            snInsList = getSnInsList(snIns)
            if 'wall' in mat and len(snInsList) == 1:
                snInsList = segFromNormal(snNormal, snInsList[0], threshold=0.95)
            # ----> Step 1: Instance level pair matching
            orInsSelect, snInsSelect = matchInsList(orInsList, snInsList)
            # ----> Step 2: align instance via optimization on iou of mask (and joint normal), also update warped uv
            maskWarpedJointOut = th.zeros_like(orInsSelect[0], requires_grad=False)
            # [1, 1, 240, 320] -> with masked out un-aligned normal pixels
            maskWarpedOut = th.zeros_like(orInsSelect[0], requires_grad=False)
            # [1, 1, 240, 320] -> also consider un-aligned normal pixels
            uvOut = th.zeros_like(orUvMap, requires_grad=False)  # [1, 2, h, w]
            warpedOrNormalOut = th.zeros_like(orNormal, requires_grad=False)  # [1, 3, h, w]
            orWarpedList = []
            warpInfoFile = cfg.warpInfoByNameId % (mat, vId)
            if not osp.exists(osp.dirname(warpInfoFile)):
                os.makedirs(osp.dirname(warpInfoFile))
            with open(warpInfoFile, 'w') as f:
                f.write('centerOrX centerOrY centerSnX centerSnY lengthOrX lengthOrY lengthSnX lengthSnY \n')
            # ---->  warping from geometry to photo:
            for idx, orIns in enumerate(orInsSelect):
                snSemMask = snInsSelect[idx]
                center1, length1 = mask2bound(orIns)
                centerOpt, lengthOpt = mask2bound(snSemMask)
                # # >>>> after finding matching, optimize for better centers and lengths
                prevC = center1
                prevL = length1
                optimizer = th.optim.Adam([center1.requires_grad_(), length1.requires_grad_()], lr=1e-4)
                orIns = orIns.float()
                _, _, H, W = orIns.shape
                xTH = 1 / W * 2
                yTH = 1 / H * 2
                for t in range(200):  # ---> use openroom object mask and scannet semantic mask to align
                    warpedOrIns = transformInp(orIns, center1, centerOpt, length1, lengthOpt)
                    loss = - th.sum(warpedOrIns * snSemMask) / th.sum(warpedOrIns + snSemMask - warpedOrIns * snSemMask)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    diffC = th.abs(prevC - center1.detach())
                    diffL = th.abs(prevL - length1.detach())
                    if diffC[0] < xTH and diffC[1] < yTH and diffL[0] < xTH and diffL[1] < yTH:
                        break
                    else:
                        prevC = center1.detach()
                        prevL = length1.detach()
                # # <<<< after finding matching, optimize for better centers and lengths
                warpedMatPart = transformInp(orMatPart.float() * orIns.float(), center1, centerOpt, length1, lengthOpt
                                             ) * snSemMask
                orWarpedList.append(warpedMatPart)
                # ---> use normal to find valid pixels
                warpedOrNormal = transformInp(orNormal, center1, centerOpt, length1, lengthOpt)
                warpedMatPartJoint = jointMask(warpedMatPart, warpedOrNormal, snNormal, normalTH)
                warpedUvMap        = transformInp(orUvMap, center1, centerOpt, length1, lengthOpt)  # 1 x 2 x H x W
                # Update accumulated output mask and uvmap
                maskWarpedJointOut = \
                    warpedMatPartJoint * warpedMatPartJoint + (1 - warpedMatPartJoint) * maskWarpedJointOut
                maskWarpedOut      = \
                    warpedMatPart      * warpedMatPart      + (1 - warpedMatPart)      * maskWarpedOut
                uvOut              = \
                    warpedMatPart      * warpedUvMap        + (1 - warpedMatPart)      * uvOut
                warpedOrNormalOut  = \
                    warpedMatPart      * warpedOrNormal     + (1 - warpedMatPart)      * warpedOrNormalOut
                with open(cfg.warpInfoByNameId % (mat, vId), 'a') as f:
                    f.write('%f %f %f %f %f %f %f %f \n' %
                        (center1[0].item(), center1[1].item(), centerOpt[0].item(), centerOpt[1].item(),
                        length1[0].item(), length1[1].item(), lengthOpt[0].item(), lengthOpt[1].item()))
                # print('warpInfo: %f %f %f %f %f %f %f %f \n' %
                #         (center1[0].item(), center1[1].item(), centerOpt[0].item(), centerOpt[1].item(),
                #         length1[0].item(), length1[1].item(), lengthOpt[0].item(), lengthOpt[1].item()))

            snMask, snMaskJoint, uvMapWarped = maskWarpedOut, maskWarpedJointOut, uvOut.permute(0, 2, 3, 1).detach()

            def postProcess(thArr):  # one dilation + two erosion
                npArr = thArr.squeeze().cpu().detach().numpy().astype(int)
                npArr = ndimage.binary_erosion( ndimage.binary_erosion( ndimage.binary_erosion(
                    ndimage.binary_dilation(npArr)))).astype(int)
                return th.from_numpy(npArr).unsqueeze(0).unsqueeze(1)
            snMask = postProcess(snMask)
            snMaskWeighted = th.sum(snNormal * warpedOrNormalOut, dim=1, keepdim=True) * snMask
            if th.sum(snMaskJoint) < 500:
                # print('--> %s Joint Mask with accurate normals is too small, only %d pixels!' % (mat, th.sum(snMask)))
                if th.sum(snMask) > 0:
                    selectedViewSmallMaskDict[mat] = selectedViewDict[mat]
                else:
                    selectedViewNoMaskDict[mat] = selectedViewDict[mat]
                    # print('Zero pixels, move to selectedNoMaskDict!')

            # Save Results:
            mm1 = Image.fromarray((orMatPart.squeeze().numpy() * 255).astype(np.uint8))
            mm1.save(cfg.maskGeoByNameId % (mat, vId))
            # Matched Results
            im = Image.fromarray( (snMask.squeeze().cpu().detach().numpy().astype(float) * 255.0).astype(np.uint8))
            im.save(cfg.maskPhotoByNameId % (mat, vId))
            snMaskJoint = snMaskJoint.long()
            im = snMaskJoint.squeeze().cpu().detach().numpy().astype(int)
            im = Image.fromarray( (im.astype(float) * 255.0).astype(np.uint8))
            im.save(cfg.maskPhotoJointByNameId % (mat, vId))
            snMaskWeighted = th.sum(snNormal * warpedOrNormalOut, dim=1, keepdim=True) * snMask
            im = snMaskWeighted.squeeze().cpu().detach().numpy()
            im = Image.fromarray( (im.astype(float) * 255.0).astype(np.uint8))
            im.save(cfg.maskPhotoWeightByNameId % (mat, vId))
            # - Raw UV
            np.save(cfg.uvWarpedByNameId % (mat, vId), uvMapWarped.cpu().numpy())
            # - visualize UV
            im = uvMapWarped.squeeze().detach().cpu().numpy()  # H x W x 2
            im = im - np.floor(im)
            im = Image.fromarray( (np.concatenate([im, np.ones((im.shape[0], im.shape[1], 1))], axis=2) * 255.0
                                   ).astype(np.uint8))
            im.save(cfg.uvWarpedVisByNameId % (mat, vId))
            im = orUvMap.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()  # H x W x 2
            im = im - np.floor(im)
            im = Image.fromarray( (np.concatenate([im, np.ones((im.shape[0], im.shape[1], 1))], axis=2) * 255.0
                                   ).astype(np.uint8))
            im.save(cfg.uvUnwarpedVisByNameId % (mat, vId))
            # Input
            target_ref = invRenderData['im']
            im = Image.fromarray( (target_ref.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(float) * 255.0
                                   ).astype(np.uint8))
            im.save(cfg.photoByNameId % (mat, vId))

            # ----> Step 2.5, warping from photo to geometry:
            # align instance via optimization on iou of mask (and joint normal), also update warped ref
            target_albedo = invRenderData['albedo']
            target_rough = invRenderData['rough']
            maskWarpedJointOut = th.zeros_like(orInsSelect[0], requires_grad=False)
            # [1, 1, 240, 320] -> with masked out un-aligned normal pixels
            maskWarpedOut = th.zeros_like(orInsSelect[0], requires_grad=False)
            # [1, 1, 240, 320] -> also consider un-aligned normal pixels
            targetOut = th.zeros_like(target_ref, requires_grad=False)  # [1, 3, h, w]
            targetAlbedoOut = th.zeros_like(target_albedo, requires_grad=False)  # [1, 3, h, w]
            targetRoughOut = th.zeros_like(target_rough, requires_grad=False)  # [1, 1, h, w]
            targetNormalOut = th.zeros_like(snNormal, requires_grad=False)  # [1, 3, h, w]
            snWarpedList = []
            for idx, orIns in enumerate(orInsSelect):
                snSemMask = snInsSelect[idx]
                center1, length1 = mask2bound(orIns)
                centerOpt, lengthOpt = mask2bound(snSemMask)
                # # >>>> after finding matching, optimize for better centers and lengths
                prevC = centerOpt
                prevL = lengthOpt
                optimizer = th.optim.Adam([centerOpt.requires_grad_(), lengthOpt.requires_grad_()], lr=1e-4)
                orIns = orIns.float()
                _, _, H, W = orIns.shape
                xTH = 1 / W * 2
                yTH = 1 / H * 2
                for t in range(200):  # ---> use openroom object mask and scannet semantic mask to align
                    warpedSnIns = transformInp(snIns.float(), centerOpt, center1, lengthOpt, length1)
                    loss = - th.sum(warpedSnIns * orIns) / th.sum(warpedSnIns + orIns - warpedSnIns * orIns)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    diffC = th.abs(prevC - centerOpt.detach())
                    diffL = th.abs(prevL - lengthOpt.detach())
                    if diffC[0] < xTH and diffC[1] < yTH and diffL[0] < xTH and diffL[1] < yTH:
                        break
                    else:
                        prevC = centerOpt.detach()
                        prevL = lengthOpt.detach()

                # print('warpInfo2: %f %f %f %f %f %f %f %f \n' %
                #         (center1[0].item(), center1[1].item(), centerOpt[0].item(), centerOpt[1].item(),
                #         length1[0].item(), length1[1].item(), lengthOpt[0].item(), lengthOpt[1].item()))

                # # <<<< after finding matching, optimize for better centers and lengths
                warpedSnMask = transformInp(snMask.float() * snSemMask.float(), centerOpt, center1, lengthOpt, length1
                                            ) * orIns
                snWarpedList.append(warpedSnMask)
                # ---> use normal to find valid pixels
                warpedSnNormal = transformInp(snNormal, centerOpt, center1, lengthOpt, length1)
                warpedSnMaskJoint = jointMask(warpedSnMask, warpedSnNormal, orNormal, normalTH)
                warpedRefImg      = transformInp(target_ref   , centerOpt, center1, lengthOpt, length1)  # 1 x 3 x H x W
                warpedAlbedo      = transformInp(target_albedo.float(), centerOpt, center1, lengthOpt, length1)
                # 1 x 3 x H x W
                warpedRough       = transformInp(target_rough.float() , centerOpt, center1, lengthOpt, length1)
                # 1 x 1 x H x W
                # Update accumulated output mask and uvmap
                maskWarpedJointOut = \
                    warpedSnMaskJoint * warpedSnMaskJoint + (1 - warpedSnMaskJoint) * maskWarpedJointOut
                maskWarpedOut      = \
                    warpedSnMask      * warpedSnMask      + (1 - warpedSnMask)      * maskWarpedOut
                targetOut          = \
                    warpedSnMask      * warpedRefImg      + (1 - warpedSnMask)      * targetOut
                targetAlbedoOut    = \
                    warpedSnMask      * warpedAlbedo      + (1 - warpedSnMask)      * targetAlbedoOut
                targetRoughOut     = \
                    warpedSnMask      * warpedRough       + (1 - warpedSnMask)      * targetRoughOut
                targetNormalOut    = \
                    warpedSnMask      * warpedSnNormal    + (1 - warpedSnMask)      * targetNormalOut

            orMaskFromSn, orMaskFromSnJoint, targetWarpedFromSn = \
                maskWarpedOut, maskWarpedJointOut, targetOut.permute(0, 2, 3, 1).detach()
            im = Image.fromarray((orMaskFromSn.squeeze().cpu().detach().numpy().astype(float) * 255.0).astype(np.uint8))
            im.save(cfg.maskGeoFromPhotoByNameId % (mat, vId))
            orMaskFromSnJoint = postProcess(orMaskFromSnJoint)
            im = orMaskFromSnJoint.squeeze().cpu().detach().numpy().astype(int)
            im = Image.fromarray((im.astype(float) * 255.0).astype(np.uint8))
            im.save(cfg.maskGeoFromPhotoJointByNameId % (mat, vId))
            orMaskFromSnWeighted = th.sum(targetNormalOut * orNormal, dim=1, keepdim=True) * orMaskFromSn
            im = orMaskFromSnWeighted.squeeze().cpu().detach().numpy()
            im = Image.fromarray((im.astype(float) * 255.0).astype(np.uint8))
            im.save(cfg.maskGeoFromPhotoWeightByNameId % (mat, vId))
            im = Image.fromarray(
                (targetWarpedFromSn.squeeze().cpu().detach().numpy().astype(float) * 255.0).astype(np.uint8))
            im.save(cfg.targetGeoFromPhotoByNameId % (mat, vId))
            targetWarped_blend = targetOut * orMatPart + 0.3 * targetOut * (~orMatPart)
            im = Image.fromarray(
                (targetWarped_blend.squeeze().permute(1, 2, 0).cpu().detach().numpy().astype(float) * 255.0).astype(
                    np.uint8))
            im.save(cfg.targetBlendGeoFromPhotoByNameId % (mat, vId))
            targetAlbedoWarpedFromSn = targetAlbedoOut.permute(0, 2, 3, 1).detach()
            im = Image.fromarray(
                (targetAlbedoWarpedFromSn.squeeze().cpu().detach().numpy().astype(float) ** (1 / 2.2) * 255.0).astype(
                    np.uint8))
            im.save(cfg.albedoGeoFromPhotoByNameId % (mat, vId))
            targetRoughWarpedFromSn = targetRoughOut.permute(0, 2, 3, 1).detach()
            im = Image.fromarray((targetRoughWarpedFromSn.squeeze().cpu().detach().numpy().astype(float) * 255.0
                                  ).astype(np.uint8))
            im.save(cfg.roughGeoFromPhotoByNameId % (mat, vId))

        for failedDict in [selectedViewSmallMaskDict, selectedViewNoMaskDict]:
            for failed in failedDict.keys():
                if failed not in selectedViewDict.keys():
                    continue
                # print('-----> Move failed material %s from selectedViewDict to selectedViewSmallMaskDict.' % failed)
                del selectedViewDict[failed]
        saveViewDict(selectedViewFile, selectedViewDict)
        saveViewDict(selectedViewSmallMaskDictFile, selectedViewSmallMaskDict)

        # Save the closest instance mask for the remaining material without any masks (semantic mask not found) #
        # Will only be used to compute median for pixels or inv rendering pred.
        for mat in bsdfIdDict.keys():
            if (mat not in selectedViewDict.keys()) and (mat not in selectedViewSmallMaskDict.keys()):
                # Find the view with largest matPartMask
                pixNumBest = 0
                viewBest = -1
                for camId in candidateCamIdList:
                    matPartMaskDict, insMaskDict = \
                        readMatPartMaskBatch(cfg.partIdPathById % int(camId),
                            bsdfIdDict, if_resize=True, imWidth=imWidth, imHeight=imHeight)
                    if mat not in matPartMaskDict.keys():  # if mat not found in current matPartMask
                        continue
                    orMatPart = th.from_numpy(matPartMaskDict[mat]).unsqueeze(0).unsqueeze(1)
                    pixNum = th.sum(orMatPart)
                    if pixNum > pixNumBest:
                        pixNumBest = pixNum
                        viewBest = camId

                if viewBest == -1:  # skip if material even not exist in OpenRoom partMask
                    # print('-----> No instance mask matched for %s. Skip!' % mat)
                    continue
                # print('-----> Saving instance mask for %s' % mat)
                camId = viewBest
                selectedViewNoMaskDict[mat] = viewBest
                # load openroom instance mask list
                matPartMaskDict, insMaskDict = \
                    readMatPartMaskBatch(cfg.partIdPathById % int(camId),
                        bsdfIdDict, if_resize=True, imWidth=imWidth, imHeight=imHeight)
                orMatPart = th.from_numpy(matPartMaskDict[mat]).unsqueeze(0).unsqueeze(1)
                orInsList = getOrInsList(mat, insMaskDict)
                if cfg.gtMaskDir is None:
                    # Read instance labels from photo
                    insLabelFile = cfg.insLabById % camId
                    insLabel  = readSegLabel(insLabelFile, if_resize=True, imWidth=imWidth, imHeight=imHeight)
                    snInsList = getSnInsList(insLabel + 1)  # collect instances with ID != 0
                else:
                    maskFile = cfg.gtMaskByNameId % (mat, camId)
                    snIns = readGtMask(maskFile, imWidth=imWidth, imHeight=imHeight, readAlpha=True)
                    snInsList = getSnInsList(snIns)

                intersectionMax = 0
                # First use hard intersection to find the mask
                for idx, snInsMask in enumerate(snInsList):
                    intersection = th.sum(orMatPart * snInsMask)
                    if intersection > intersectionMax:
                        intersectionMax = intersection
                        snMask = snInsMask
                # If not found, use soft mIoU to find the mask
                if intersectionMax == 0:
                    mIoUMax = 0
                    for idx, snInsMask in enumerate(snInsList):
                        orInsMaskSoft = toSoftMask(orMatPart)
                        snInsMaskSoft = toSoftMask(snInsMask)
                        intersection = th.sum(orInsMaskSoft * snInsMaskSoft)
                        union = th.sum(orInsMaskSoft + snInsMaskSoft)
                        mIoU = intersection / union
                        if mIoU > mIoUMax and mIoU > 0:
                            mIoUMax = mIoU
                            snMask = snInsMask

                im = Image.fromarray( (snMask.squeeze().cpu().detach().numpy().astype(float) * 255.0).astype(np.uint8))
                os.makedirs(os.path.dirname(cfg.maskPhotoByNameId % (mat, camId)), exist_ok=True)
                im.save(cfg.maskPhotoByNameId % (mat, camId))

        saveViewDict(selectedViewNoMaskDictFile, selectedViewNoMaskDict)

        selectedViewFullDict = {}
        for svDict in [selectedViewDict, selectedViewSmallMaskDict, selectedViewNoMaskDict]:
            for k, v in svDict.items():
                selectedViewFullDict[k] = v
        saveViewDict(selectedViewFullDictFile, selectedViewFullDict)
    else:
        selectedViewDict = readViewDict(selectedViewFile)
        selectedViewSmallMaskDict = readViewDict(selectedViewSmallMaskDictFile)
        selectedViewNoMaskDict = readViewDict(selectedViewNoMaskDictFile)
        selectedViewFullDict = readViewDict(selectedViewFullDictFile)

    return selectedViewDict, selectedViewSmallMaskDict, selectedViewFullDict


def saveSelectedCamAndId(matViewDict, selectedCamFile, selectedCamIdFile, newCamDir):
    vNum = len(matViewDict.keys())
    vIdList = []
    with open(selectedCamFile, 'w') as cam:
        cam.write('%d\n' % vNum)
        for mat, vId in matViewDict.items():
            vIdList.append(vId)
            camFile = osp.join(newCamDir, '%d.txt' % vId)
            camPoseDict = readCamPoseDictList(camFile)[0]
            origin = camPoseDict['origin']
            target = camPoseDict['target']
            up     = camPoseDict['up']
            cam.write('{} {} {}\n'.format(origin[0], origin[1], origin[2]))
            cam.write('{} {} {}\n'.format(target[0], target[1], target[2]))
            cam.write('{} {} {}\n'.format(up[0], up[1], up[2]))
    # print('New cam file for selected views is saved at %s' % selectedCamFile)
    with open(selectedCamIdFile, 'w') as cam:
        cam.writelines('%d ' % camId for camId in vIdList)
    # print('New camId file for selected views is saved at %s' % selectedCamIdFile)
    return vIdList


def saveCandidateCamFile(vIdList, outCamFile, newCamDir):
    with open(outCamFile, 'w') as cam:
        cam.write('%d\n' % len(vIdList))
        for vId in vIdList:
            camFile = osp.join(newCamDir, '%d.txt' % vId)
            camPoseDict = readCamPoseDictList(camFile)[0]
            origin = camPoseDict['origin']
            target = camPoseDict['target']
            up     = camPoseDict['up']
            cam.write('{} {} {}\n'.format(origin[0], origin[1], origin[2]))
            cam.write('{} {} {}\n'.format(target[0], target[1], target[2]))
            cam.write('{} {} {}\n'.format(up[0], up[1], up[2]))
    print('New cam file for selected views is saved at %s' % outCamFile)


def copySelectedPhoto(selectedViewIdList, photoDir, photoSrcDir, ext='png'):
    if not osp.exists(photoDir):
        os.system('mkdir -p %s' % photoDir)
    for vId in selectedViewIdList:
        srcImg = osp.join(photoSrcDir, '%d.%s' % (vId, ext))
        cmd = 'cp %s %s' % (srcImg, photoDir)
        os.system(cmd)


def saveSelectedCam(camPoseDict, selectedCamFile):
    vNum = 1
    with open(selectedCamFile, 'w') as cam:
        cam.write('%d\n' % vNum)
        cam.write('%s\n' % camPoseDict['origin'])
        cam.write('%s\n' % camPoseDict['target'])
        cam.write('%s\n' % camPoseDict['up'])
    print('New cam file for selected views is saved at %s' % selectedCamFile)


def readCamPoseDictList(newCamFile):
    lc = 0
    camPoseDictList = []
    with open(newCamFile, 'r') as f:
        for line in f.readlines():
            lc += 1
            if lc == 1:
                # cNum = int(line.strip())
                camPoseDict = {}
            else:
                cVal = line.strip().split(' ')
                cValArr = np.array([float(cVal[0]), float(cVal[1]), float(cVal[2])])
                if lc % 3 == 2:
                    camPoseDict['origin'] = cValArr
                elif lc % 3 == 0:
                    camPoseDict['target'] = cValArr
                elif lc % 3 == 1:
                    camPoseDict['up']     = cValArr
                    camPoseDictList.append(camPoseDict)
                    camPoseDict = {}
    # print('Cam pose dict is read from %s' % newCamFile)
    return camPoseDictList


def updateXMLFromTotal3D(cfg):
    def saveLayoutMesh(objInfoDict, matInfoDict, layoutArr, meshSaveDir, blenderApplyUvCmdPre, shapeCntInit=0,
            bsdfCntInit=0):
        def vToLine(v):
            return 'v %f %f %f\n' % (v[0], v[1], v[2])

        def newObjInfo(shapeName, objPath, matList):
            objInfo = {'shape_id': shapeName, 'objPath': objPath, 'matList': matList,
                    'scale': np.array([1.0, 1.0, 1.0]), 'isEmitter': False, 'emitterVal': None,
                    'transform_opList': ['scale'],
                    'transform_attribList': [{'x': '1.000000', 'y': '1.000000', 'z': '1.000000'}],
                    'isAlignedLight': False, 'isContainer': False}
            return objInfo

        def newWhiteMatInfo(bsdfName):
            matInfo = {'bsdf_id': bsdfName, 'isHomo': True, 'albedo': '0.900 0.900 0.900', 'roughness': '0.700',
                    'albedoScale': '1.000 1.000 1.000', 'roughnessScale': '1.000', 'uvScale': '1.0'}
            return matInfo

        vc1, vc2, vc3, vc4 = layoutArr[0], layoutArr[1], layoutArr[2], layoutArr[3]
        vf1, vf2, vf3, vf4 = layoutArr[4], layoutArr[5], layoutArr[6], layoutArr[7]

        layoutMeshSaveDir = meshSaveDir
        if not osp.exists(layoutMeshSaveDir):
            os.system('mkdir -p %s' % layoutMeshSaveDir)

        shapeCnt, bsdfCnt = shapeCntInit, bsdfCntInit

        # floor
        floorInitPath = osp.join(layoutMeshSaveDir, 'scene_floor_init.obj')
        floorPath = osp.join(layoutMeshSaveDir, 'scene_floor.obj')
        with open(floorInitPath, 'w') as f:
            for v in [vf1, vf2, vf3, vf4]:
                f.write(vToLine(v))
            f.write('usemtl 2_floor_part0\n')
            f.write('f %d %d %d\n' % (1, 2, 3))
            f.write('f %d %d %d\n' % (3, 4, 1))
        os.system(blenderApplyUvCmdPre % (floorInitPath, floorPath))
        print('Floor mesh saved at %s' % floorPath)
        objInfoDict[shapeCnt] = newObjInfo('2_floor_object', floorPath, ['2_floor_part0'])
        shapeCnt += 1
        matInfoDict[bsdfCnt] = newWhiteMatInfo('2_floor_part0')
        bsdfCnt += 1

        # ceiling
        ceilingInitPath = osp.join(layoutMeshSaveDir, 'scene_ceiling_init.obj')
        ceilingPath = osp.join(layoutMeshSaveDir, 'scene_ceiling.obj')
        with open(ceilingInitPath, 'w') as f:
            for v in [vc1, vc2, vc3, vc4]:
                f.write(vToLine(v))
            f.write('usemtl 22_ceiling_part0\n')
            f.write('f %d %d %d\n' % (1, 2, 3))
            f.write('f %d %d %d\n' % (3, 4, 1))
        os.system(blenderApplyUvCmdPre % (ceilingInitPath, ceilingPath))
        print('Ceiling mesh saved at %s' % ceilingPath)
        objInfoDict[shapeCnt] = newObjInfo('22_ceiling_object', ceilingPath, ['22_ceiling_part0'])
        shapeCnt += 1
        matInfoDict[bsdfCnt] = newWhiteMatInfo('22_ceiling_part0')
        bsdfCnt += 1

        # wall
        wallInitPath = osp.join(layoutMeshSaveDir, 'scene_wall_init.obj')
        wallPath = osp.join(layoutMeshSaveDir, 'scene_wall.obj')
        with open(wallInitPath, 'w') as f:
            for v in [vc1, vc2, vc3, vc4, vf1, vf2, vf3, vf4]:
                f.write(vToLine(v))
            f.write('usemtl 1_wall_part0\n')
            f.write('f %d %d %d\n' % (1, 5, 6))
            f.write('f %d %d %d\n' % (6, 2, 1))
            f.write('usemtl 1_wall_part1\n')
            f.write('f %d %d %d\n' % (2, 6, 7))
            f.write('f %d %d %d\n' % (7, 3, 2))
            f.write('usemtl 1_wall_part2\n')
            f.write('f %d %d %d\n' % (3, 7, 8))
            f.write('f %d %d %d\n' % (8, 4, 3))
            f.write('usemtl 1_wall_part3\n')
            f.write('f %d %d %d\n' % (4, 8, 5))
            f.write('f %d %d %d\n' % (5, 1, 4))
        os.system(blenderApplyUvCmdPre % (wallInitPath, wallPath))
        print('Wall mesh saved at %s' % wallPath)
        objInfoDict[shapeCnt] = newObjInfo('1_wall_object', wallPath,
            ['1_wall_part0', '1_wall_part1', '1_wall_part2', '1_wall_part3'])
        shapeCnt += 1
        matInfoDict[bsdfCnt] = newWhiteMatInfo('1_wall_part0')
        bsdfCnt += 1
        matInfoDict[bsdfCnt] = newWhiteMatInfo('1_wall_part1')
        bsdfCnt += 1
        matInfoDict[bsdfCnt] = newWhiteMatInfo('1_wall_part2')
        bsdfCnt += 1
        matInfoDict[bsdfCnt] = newWhiteMatInfo('1_wall_part3')
        bsdfCnt += 1

        return shapeCnt, bsdfCnt

    def saveObjectMesh(objInfoDict, matInfoDict, objDictList, total3dOutputDir, meshSaveDir, blenderApplyUvCmdPre,
            shapeCntInit=0, bsdfCntInit=0):
        def newObjInfo(shapeName, objPath, matList, opList, attribList):
            assert (len(opList) == len(attribList))
            objInfo = {'shape_id': shapeName, 'objPath': objPath, 'matList': matList,
                    'scale': np.array([1.0, 1.0, 1.0]), 'isEmitter': False, 'emitterVal': None,
                    'transform_opList': opList, 'transform_attribList': attribList,
                    'isAlignedLight': False, 'isContainer': False}
            return objInfo

        def newWhiteMatInfo(bsdfName):
            matInfo = {'bsdf_id': bsdfName, 'isHomo': True, 'albedo': '0.900 0.900 0.900', 'roughness': '0.700',
                    'albedoScale': '1.000 1.000 1.000', 'roughnessScale': '1.000', 'uvScale': '1.0'}
            return matInfo

        def getVert(objSrc):
            vList = []
            with open(objSrc, 'r') as f:
                for line in f.readlines():
                    items = line.strip().split()
                    if items[0] == 'v':
                        vList.append(np.array([float(items[1]), float(items[2]), float(items[3])]))
            vArr = np.stack(vList, axis=0)
            return vArr

        def getMinMax(vArr):
            return np.amin(vArr, axis=0), np.amax(vArr, axis=0)

        shapeCnt, bsdfCnt = shapeCntInit, bsdfCntInit
        for objIdx, objDict in enumerate(objDictList):
            basis, coeffs, centroid, classId = objDict[0][0]  # basis, coeffs, centroid, classId
            objSrc = osp.join(total3dOutputDir, '%d_%d.obj' % (objIdx, classId[0]))
            objUV = osp.join(meshSaveDir, osp.basename(objSrc))
            os.system(blenderApplyUvCmdPre % (objSrc, objUV))
            bsdfName = '%d_%d_part0' % (classId[0], objIdx)
            modifyMaterialName(objUV, bsdfName)
            print('New mesh with UV saved at %s' % objUV)
            # Mesh vertex min and max
            vArr = getVert(objSrc)
            meshMin, meshMax = getMinMax(vArr)
            meshCenter = 0.5 * (meshMin + meshMax)
            vArr = vArr - meshCenter
            meshMin, meshMax = getMinMax(vArr)
            meshScale = 0.5 * (meshMax - meshMin)
            scale = coeffs[0] / meshScale
            # Compute rotation angles
            R = np.transpose(basis)
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            if not singular :
                x = math.atan2(R[2, 1] , R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0

            opList = ['translate', 'scale', 'rotate', 'rotate', 'rotate', 'translate']
            attribList = [ {'x': str(-meshCenter[0]), 'y': str(-meshCenter[1]), 'z': str(-meshCenter[2])},
                            {'x': str(scale[0]), 'y': str(scale[1]), 'z': str(scale[2])},
                            {'angle': str(np.rad2deg(x)), 'x': '1.0', 'y': '0.0', 'z': '0.0'},
                            {'angle': str(np.rad2deg(y)), 'x': '0.0', 'y': '1.0', 'z': '0.0'},
                            {'angle': str(np.rad2deg(z)), 'x': '0.0', 'y': '0.0', 'z': '1.0'},
                            {'x': str(centroid[0][0]), 'y': str(centroid[0][1]), 'z': str(centroid[0][2])}]
            objInfoDict[shapeCnt]\
                = newObjInfo('%d_%d_object' % (classId[0], objIdx), objUV, [bsdfName], opList, attribList)
            shapeCnt += 1
            matInfoDict[bsdfCnt] = newWhiteMatInfo(bsdfName)
            bsdfCnt += 1
        return shapeCnt, bsdfCnt

    def saveCeilingAreaLightMesh(layoutArr, meshSaveDir, lDist=3, lSizeRatio=0.1):
        vArr = layoutArr[0:4, :]
        verts = th.from_numpy(vArr).type(th.FloatTensor)  # nVert x 3
        U, S, V = th.pca_lowrank(verts, q=None, center=True, niter=2)
        axis1, axis2 = V[:, 0], V[:, 1]
        axis1 = axis1 / th.norm(axis1)
        axis2 = axis2 / th.norm(axis2)
        vP1 = th.sum(verts * axis1.unsqueeze(0), dim=1)  # proj to first principal axis
        length1 = (th.max(vP1) - th.min(vP1)) * 0.9
        vP2 = th.sum(verts * axis2.unsqueeze(0), dim=1)  # proj to second principal axis
        length2 = (th.max(vP2) - th.min(vP2)) * 0.9
        center = (th.max(verts, dim=0)[0] + th.min(verts, dim=0)[0]) / 2.0
        nLight1 = th.div(length1 + 0.5 * lDist, lDist, rounding_mode='floor')
        nLight1 = nLight1 if nLight1 >= 1 else 1
        nLight2 = th.div(length2 + 0.5 * lDist, lDist, rounding_mode='floor')
        nLight2 = nLight2 if nLight2 >= 1 else 1
        corner = center - 0.5 * length1 * axis1 - 0.5 * length2 * axis2
        step1 = length1 / nLight1
        step2 = length2 / nLight2
        lSize1 = step1 * lSizeRatio
        lSize2 = step2 * lSizeRatio

        emitterList = []
        lightCnt = 0
        for n1 in range(int(nLight1)):
            for n2 in range(int(nLight2)):
                newVList = []
                newFList = []
                pos = corner + (0.5 + n1) * step1 * axis1 + (0.5 + n2) * step2 * axis2
                v1 = pos + lSize1 * axis1 + lSize2 * axis2
                v2 = pos + lSize1 * axis1 - lSize2 * axis2
                v3 = pos - lSize1 * axis1 - lSize2 * axis2
                v4 = pos - lSize1 * axis1 + lSize2 * axis2
                newVList += [v1, v2, v3, v4]
                newFList.append(np.array([1, 2, 3]))
                newFList.append(np.array([1, 3, 4]))

                saveRoot = osp.join(meshSaveDir, '%s_light%d.obj' % ('scene', lightCnt))
                if not osp.exists(osp.dirname(saveRoot)):
                    os.system('mkdir -p %s' % osp.dirname(saveRoot))
                with open(saveRoot, 'w') as f:
                    f.write('')
                # write v
                for idx, vv in enumerate(newVList):
                    line = 'v %f %f %f' % (vv[0], vv[1], vv[2] - 0.001)
                    with open(saveRoot, 'a') as f:
                        f.write('{}\n'.format(line))

                # write f
                for face in newFList:
                    v1, v2, v3 = face[0], face[1], face[2]
                    line = 'f %d %d %d' % (v1, v2, v3)
                    with open(saveRoot, 'a') as f:
                        f.write('{}\n'.format(line))
                print('New light mesh saved at %s' % saveRoot)
                lightCnt += 1

                emitterList.append({'objPath': saveRoot, 'area': lSize1 * lSize2})

        return emitterList

    def processInputPhoto(rgbImg, photoSavePath, intrinsicMat):
        # Process input photo #
        img = rgbImg
        H, W, _ = img.shape
        if W > (H * 4 / 3):  # clip x
            res = int((W - (H * 4 / 3)) // 2)
            img = img[:, res:W - res, :]
        elif W < (H * 4 / 3):  # clip y
            res = int((H - (W * 3 / 4)) // 2)
            img = img[res:H - res, :, :]
        im = Image.fromarray(img)
        if not osp.exists(osp.dirname(photoSavePath)):
            os.makedirs(osp.dirname(photoSavePath))
        im.save(photoSavePath)
        HNew, WNew, CNew = img.shape

        focusX, prinX = intrinsicMat[0, 0], intrinsicMat[0, 2]
        # focusY, prinY = intrinsicMat[1, 1], intrinsicMat[1, 2]
        # tan = float(prinX) / (0.5 * float(focusX) + 0.5 * float(focusY))
        tan = float(prinX) / float(focusX)
        fovX = np.rad2deg(np.arctan(WNew * tan / W)) * 2

        return fovX

    def processCamera(rot_cam):

        origin = np.array([0, 0, 0])
        target = np.matmul(rot_cam, np.array([1, 0, 0]))
        up = np.matmul(rot_cam, np.array([0, 1, 0]))

        camInfoDict = {}
        camInfoDict['target'] = '%f %f %f' % (target[0], target[1], target[2])
        camInfoDict['origin'] = '%f %f %f' % (origin[0], origin[1], origin[2])
        camInfoDict['up']     = '%f %f %f' % (up[0]    , up[1]    , up[2])

        return camInfoDict

    # Total3D Input
    with open(cfg.total3dInputFile, 'rb') as f:
        data = pickle.load(f)
    rgbImg           = data['rgb_img']
    intrinsicMat     = data['camera']['K']
    # Total3D Output
    layoutFile       = osp.join(cfg.total3dOutputDir, 'layout.mat')  # Input
    bdbFile          = osp.join(cfg.total3dOutputDir, 'bdb_3d.mat')  # Input
    extrinsicFile    = osp.join(cfg.total3dOutputDir, 'r_ex.mat')  # Input

    layoutArr        = scipy.io.loadmat(layoutFile)['layout']  # 8 x 3, ceiling: lu, ld, rd, ru / floor: lu, ld, rd, ru
    rot_cam          = scipy.io.loadmat(extrinsicFile)['cam_R']  # 3 x 3
    objDictList      = scipy.io.loadmat(bdbFile)['bdb'][0]  # number of objects

    objInfoDict, matInfoDict = {}, {}
    shapeCnt, bsdfCnt = saveLayoutMesh(objInfoDict, matInfoDict, layoutArr, cfg.meshSrcSaveDir,
        cfg.blenderApplyUvCmdPre)
    shapeCnt, bsdfCnt = saveObjectMesh(objInfoDict, matInfoDict, objDictList, cfg.total3dOutputDir,
        cfg.meshSrcSaveDir, cfg.blenderApplyUvCmdPre, shapeCntInit=shapeCnt, bsdfCntInit=bsdfCnt)
    emitterList = saveCeilingAreaLightMesh(layoutArr, cfg.meshSrcSaveDir)

    fovX = processInputPhoto(rgbImg, cfg.photoSavePath, intrinsicMat)

    camInfoDict = processCamera(rot_cam)

    cv2.imwrite(cfg.envFile, np.ones((240, 320, 3)) * 100)

    sensorInfoDict = {'fov': str(fovX), 'fovAxis': 'x', 'width': '320', 'height': '240',
                        'sampler_type': 'adaptive', 'sample_count': '128'}
    envInfoDict = {'envPath': cfg.envFile, 'scale': '1.0'}
    xmlInfoDict = {'shape': objInfoDict, 'bsdf': matInfoDict, 'sensor': sensorInfoDict, 'emitter': envInfoDict}
    _ = createXmlFromInfoDict(cfg.xmlSrcFile, xmlInfoDict)

    saveSelectedCam(camInfoDict, cfg.srcCamFile)

    return [camInfoDict], emitterList


def saveCamFile(newCamFile, newCamIdFile, newCamDir, camIdList, camPoseDictList):
    if not osp.exists(osp.dirname(newCamFile)):
        os.makedirs(osp.dirname(newCamFile))
    with open(newCamFile, 'w') as cam:
        cam.write('%d\n' % len(camPoseDictList))

    if not osp.exists(newCamDir):
        os.makedirs(newCamDir)
    for idx, camPoseDict in enumerate(camPoseDictList):
        origin = camPoseDict['origin']
        target = camPoseDict['target']
        up     = camPoseDict['up']
        # append joint cam file
        with open(newCamFile, 'a') as cam:
            cam.write('{}\n'.format(origin))
            cam.write('{}\n'.format(target))
            cam.write('{}\n'.format(up))

        camId = camIdList[idx]
        singleCamFile = osp.join(newCamDir, '%s.txt' % camId)
        if not osp.exists(newCamDir):
            os.makedirs(newCamDir)
        # save separate single cam file
        with open(singleCamFile, 'w') as cam:
            cam.write('1\n')
            cam.write('{}\n'.format(origin))
            cam.write('{}\n'.format(target))
            cam.write('{}\n'.format(up))

        # save cam id file
        with open(newCamIdFile, 'a') as cam:
            cam.write('%s ' % camId)

    print('New cam file is saved at %s' % newCamFile)
    print('New camId file is saved at %s' % newCamIdFile)


def readLabelMapDict(labelMapFile):
    labelMapDict = {}
    with open(labelMapFile, 'r') as f:
        for line in f.readlines():
            orLabel, snStrList = line.strip().split(' ')
            snList = snStrList.split(',')
            snList = [int(sId) for sId in snList]
            labelMapDict[orLabel] = snList
    return labelMapDict


def renderSceneLabels(cfg):
    # Render Part Label
    renderPass(cfg, cfg.initRenderDir, cfg.partIdItem, cfg.xmlFile, cfg.allCamFile, cfg.camIdList)
    # Render Normal
    renderPass(cfg, cfg.initRenderDir, cfg.normalItem, cfg.xmlFile, cfg.allCamFile, cfg.camIdList)
    # Render UV
    renderPass(cfg, cfg.initRenderDir, cfg.uvItem    , cfg.xmlFile, cfg.allCamFile, cfg.camIdList)
    # Render Depth
    renderPass(cfg, cfg.initRenderDir, cfg.depthItem , cfg.xmlFile, cfg.allCamFile, cfg.camIdList)


def sampleView(sampledViewFile, camIdList, newCamDir, baselineAngle=30, baselineDist=1, isOverwrite=False):
    if isOverwrite or not osp.exists(sampledViewFile):
        # >>>> Select candidate cams via baseline angle and baseline distance
        print('Selcting candidate views via baseline angles ...')
        candidateCamIdList = []
        candidateLookAtList = []
        candidateOriginList = []
        for cc in tqdm(range(len(camIdList))):
            camId = int(camIdList[cc])
            # Read Cam Pose Look At direction
            camFile = osp.join(newCamDir, '%d.txt' % camId)
            camPoseDict = readCamPoseDictList(camFile)[0]
            origin = camPoseDict['origin']
            target = camPoseDict['target']
            lookAt = target - origin
            lookAt = lookAt / np.linalg.norm(lookAt)
            if camId == 0:
                candidateCamIdList.append(camId)
                candidateLookAtList.append(lookAt)
                candidateOriginList.append(origin)
            else:
                def checkSimilar():
                    for idx, og in enumerate(candidateOriginList):
                        la = candidateLookAtList[idx]
                        cond1 = np.arccos(np.vdot(lookAt, la)) / math.pi * 180 < baselineAngle
                        cond2 = np.linalg.norm(origin - og) < baselineDist
                        if cond1 and cond2:
                            return True
                    return False
                if checkSimilar():
                    continue
                candidateCamIdList.append(camId)
                candidateLookAtList.append(lookAt)
                candidateOriginList.append(origin)

        saveViewList(candidateCamIdList, sampledViewFile)
        # <<<< Select candidate cams via baseline angle and baseline distance
    else:
        candidateCamIdList = readViewList(sampledViewFile)
    print('Sampled %d views:' % len(candidateCamIdList), end='')
    for cc in candidateCamIdList:
        print(' %d' % cc, end='')
    print('')
    return candidateCamIdList


def genImListForInit(camIdList, initPerPixelImListFile, ext='png'):
    with open(initPerPixelImListFile, 'w') as f:
        for cId in camIdList:
            f.writelines('%s.%s\n' % (cId, ext))


if __name__ == '__main__':
    # print('\n')
    parser = argparse.ArgumentParser(description='Preprocess Script for PhotoScene')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # SUN RGB-D Inputs #
    if cfg.dataMode == 'total3d':  # total 3D

        # Generate Total 3D Inputs
        os.system(cfg.total3dPreprocessCmd)  # convert .pkl to demo folder

        # evaluate with total3d
        print('\n---> Running Total 3D ... (This may take a while)')
        os.system(cfg.total3dRunCmd)

        # Generate main.xml file for scene configurations
        print('\n---> Initializing scene meshes (with UVs) and configurations...')
        camPoseDictList, emitterList = updateXMLFromTotal3D(cfg)

        # Generate camera file
        cfg.camIdList       = ['0']
        saveCamFile(cfg.allCamFile, cfg.allCamIdFile, cfg.allCamDir, cfg.camIdList, camPoseDictList)

        # generate panoptic label
        print('\n---> Running MaskFormer to get panoptic labels...')
        os.system(cfg.genPanopticCmd)

    else:
        assert (False)

    labelMapDict    = readLabelMapDict(cfg.labelMapFile)

    # Save new models.txt file to each scene, and
    # generate a new xmlfile to enable diff. materials assigned to same shape
    print('\n---> Generating new xml files and model ID file and computing initial UV scale ...')
    _, _ = saveNewModelIdFileAndXmlFile(
        cfg.xmlSrcFile, cfg.meshSaveDir, modelIdFileNew=cfg.modelIdFile, xmlFileNew=cfg.xmlFile)
    _ = saveNewWhiteXML(cfg.xmlFile, imWidth=cfg.initImgWidth, imHeight=cfg.initImgHeight, outPath=cfg.xmlWhiteFile)

    # save a new xml file with white materials
    print('\n---> Rendering the initial scene labels ...')
    renderSceneLabels(cfg)

    # >>> get mat part id dict
    modelStrIdDict = readModelStrIdDict(cfg.modelIdFile)
    bsdfIdDict = getBsdfIdDict(cfg.xmlFile, modelStrIdDict)
    # <<< get mat part id dict

    # >>> sub-sample views, for multi-view (video) input, e.g. ScanNet-to-OpenRooms
    print('\n---> Sub-sampling views from total %d views...' % (len(cfg.camIdList)))
    sampledIdList = sampleView(cfg.sampledCamIdFile, cfg.camIdList, cfg.allCamDir,
        isOverwrite=cfg.isOverWriteSampledView)
    saveCandidateCamFile(sampledIdList, cfg.sampledCamFile, cfg.allCamDir)
    copySelectedPhoto(sampledIdList, cfg.photoDir, cfg.photoSrcDir)
    # >>> sub-sample views

    # >>>>> generate camId list for inverse rendering initialization and material part segments
    print('\n---> Generating inverse rendering initailization... (This may take a while)')
    genImListForInit(sampledIdList, cfg.initPerPixelImListFile)
    os.system(cfg.invrenderCmd)

    # >>> view selection and save ground truth openroom material masks for selected views, use semantic labels!
    print('\n---> Searching for optimal view for each material and saving preprocessed results ...')
    selectedViewDict, selectedViewFailedDict, selectedViewFullDict \
        = selectViewAndSaveMask(cfg, sampledIdList, bsdfIdDict,
                imWidth=cfg.imgWidth, imHeight=cfg.imgHeight,
                labelMapDict=labelMapDict,
                isOverwrite=cfg.isOverWriteSelectedView)
    # write cam and camId files for the selected views
    selectedIdList = saveSelectedCamAndId(selectedViewDict, cfg.selectedCamFile, cfg.selectedCamIdFile, cfg.allCamDir)
    # <<< view selection

    _ = saveNewBaselineXML(cfg, baseline='median')

    # Save Single Light XML Files from main_median.xml
    if cfg.dataMode == 'total3d':
        genSingleLightAreaXMLs(cfg, emitterList)
    else:
        assert (False)

    print('\n---> Preprocessing is done!\n\n')
