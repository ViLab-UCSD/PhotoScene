from __future__ import print_function
import os
import argparse
import random
import os.path as osp
import cv2
import numpy as np
import datetime
from utils.rendering_layer import MicrofacetUV
import torch as th
from PIL import Image
import math
from tqdm import tqdm

from utils.io import *
from utils.xml import *


def loadBatchMaterialData(graphSampleResultDirList, useHomoRough=False, device='cuda'):
    matDict = {}
    for m in ['basecolor', 'normal', 'roughness']:
        imOriList = []
        for graphSampleResultDir in graphSampleResultDirList:
            imFile = os.path.join(graphSampleResultDir, m, '%s.png' % m)
            imOri      = Image.open(imFile).convert('RGB')
            imOri      = np.asarray(imOri, dtype=np.float32) / 255.0
            if m == 'basecolor':
                imOri = imOri ** 2.2
            elif m == 'normal':
                imOri = (imOri - 0.5) * 2.0
            else:
                imOri = imOri
            imOri = th.from_numpy(imOri).to(device).permute(2, 0, 1).unsqueeze(0)
            if useHomoRough and m == 'roughness':
                roughHomo = th.median(imOri)
                imOri = roughHomo.repeat(imOri.shape)
            imOriList.append(imOri)
        matDict[m] = th.cat(imOriList, dim=0) # b x 3 x refH x refW

    return matDict


def computeError(pos_render, neg_optim_render, optObjectiveDict, renderObj):
    
    err_stat = renderObj.applyMaskStatLoss(neg_optim_render, pos_render)
    coeff_stat = optObjectiveDict['stat'] if 'stat' in optObjectiveDict.keys() else th.zeros_like(err_stat)
    
    err_pix = renderObj.applyMaskLoss(neg_optim_render, pos_render)
    coeff_pix = optObjectiveDict['pix'] if 'pix' in optObjectiveDict.keys() else th.zeros_like(err_pix)
    
    err_vgg = renderObj.applyMaskVggLoss(neg_optim_render, pos_render)
    coeff_vgg = optObjectiveDict['vgg'] if 'vgg' in optObjectiveDict.keys() else th.zeros_like(err_vgg)
    
    err = err_stat * coeff_stat + err_pix * coeff_pix + err_vgg * coeff_vgg
    return err, err_stat, err_pix, err_vgg


def adjustColor(source, target, mask):
    # Align median color
    # sample_rendered_img: B x C x H x W
    # pos_render: 1 x C x H x W
    target_linear = target ** 2.2
    source_linear = source ** 2.2
    mask = mask > 0.9
    source_median = th.ones((source.shape[0], 3, 1, 1))
    for b in range(source.shape[0]):
        for c in range(3):
            source_median[b, c, 0, 0] = th.median(source_linear[b, c, :, :].view(-1)[mask.view(-1)])

    target_median = th.ones((1, 3, 1, 1))
    for c in range(3):
        target_median[0, c, 0, 0] = th.median(target_linear[0, c, :, :].view(-1)[mask.view(-1)])
    return (source_linear / source_median * target_median).clamp(0, 1) ** (1/2.2)
    # return (source_linear - source_median + target_median).clamp(0, 1) ** (1/2.2)


def toGrayscale(img):
    # N x C x H x W
    img_linear = th.where(img <= 0.04045, img / 12.92, ( (img + 0.055) / 1.055 ) ** 2.4)
    gray_linear = 0.2126 * img_linear[:, 0, :, :] + 0.7152 * img_linear[:, 1, :, :] + 0.0722 * img_linear[:, 2, :, :]
    gray_srgb = th.where(gray_linear <= 0.0031308, gray_linear * 12.92, gray_linear ** (1/2.4) * 1.055 - 0.055)
    return gray_srgb.unsqueeze(1).repeat(1, 3, 1, 1)


def saveTensorImg(tensor, savePathList):
    assert(len(savePathList) == tensor.shape[0])
    # tensor: N x C x H x W
    for idx, path in enumerate(savePathList):
        if not osp.exists(osp.dirname(path)):
            os.system('mkdir -p %s' % osp.dirname(path))
        im = Image.fromarray((tensor[idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        im.save(path)
        print('Image saved at %s' % path)


def main(cfg):
    
    matPartList = []
    vIdList = []
    with open(cfg.selectedViewFile, 'r') as fIn:
        for x in fIn.readlines():
            matPartName, vId = x.strip().split(' ')
            matPartList.append(matPartName)
            vIdList.append(vId)

    if cfg.seed is not None:
        random.seed(cfg.seed)
    else:
        seed = random.randint(0, 1000)
        random.seed(seed)
        print('Selected seed is %d!' % seed)
        now = datetime.datetime.now()
        with open(cfg.seedFile, 'a') as f:
            f.write(now.strftime("%Y-%m-%d %H:%M:%S") + ' selectedGraphSeed: %d\n' % seed)

    if cfg.usekNN:
        graphList = []
        with open(cfg.graphListFile, 'r') as f:
            for line in f.readlines():
                graphList.append(line.strip())

        optObjectiveDict = {'vgg': 1.0} # only need vgg loss to look for similar texture patterns

        refH, refW, res = cfg.imgHeight, cfg.imgWidth, 2**cfg.matRes
        if th.cuda.is_available():
            device_full_name = 'cuda'
            device = th.device(device_full_name)
            th.set_default_tensor_type('torch.cuda.FloatTensor')
            print('Use GPU')
        else:
            device_full_name = 'cpu'
            device = th.device(device_full_name)
            th.set_default_tensor_type('torch.FloatTensor')
            print('Use CPU')

        fov = float(getSensorInfoDict(cfg.xmlFile)['fov'])
        with open(cfg.selectedGraphFile, 'w') as f:
            for mId, matName in enumerate(matPartList):
                print('\n---> Searching for optimal procedural graph for %s ...' % matName)
                # check # of mask pixel, if too small, assign homo graph
                maskPath = cfg.maskPhotoByNameId % (matName, int(vIdList[mId]))
                pixNum = np.sum(cv2.imread(maskPath, -1) / 255.0)
                if pixNum < 1000:
                    selectedGraph = 'homogeneous'
                    print('Selected homogeneous graph!')
                else:
                    # Search for nearest neighbors over masked region between rendering and target
                    vId = int(vIdList[mId])
                    inputData = loadOptInput(cfg, matName, vId, refH, refW, mode='photo', device=cfg.device)

                    batchSize = cfg.graphSelectBatchSize
                    renderObj = MicrofacetUV(res, inputData, \
                                    imHeight=refH, imWidth=refW, fov=fov, \
                                    useVgg=True, useStyle=False, onlyMean=False, isHdr=False, device=device)
                    
                    pos_render  = inputData['im']
                    if cfg.graphSelectToGray:
                        pos_render  = toGrayscale(pos_render)
                    
                    graphSampleResultDirList = []
                    graphLabelList = []
                    for graphIdx, graph in enumerate(graphList):
                        for sampleIdx in range(cfg.nSamplePerGraph):
                            graphSampleResultDir = os.path.join(cfg.graphSampleDir, graph, str(sampleIdx))
                            graphSampleResultDirList.append(graphSampleResultDir)
                            graphLabelList.append(graphIdx)
                    
                    ##### Load Sampled Graph Materials #####
                    errList = []
                    nBatch = math.ceil(len(graphSampleResultDirList) / batchSize) 
                    for batchIdx in tqdm(range(nBatch)):
                        start = batchIdx*batchSize
                        end = min((batchIdx+1)*batchSize, len(graphSampleResultDirList))
                        resultDirList = graphSampleResultDirList[start:end]
                        mData = loadBatchMaterialData(resultDirList, useHomoRough=cfg.useHomoRough, device=device)

                        errAll = []
                        uvScaleBase = 1.5
                        angBase, angStep = math.pi / 4, cfg.nUvAngStep # 45 deg
                        uvScaleLogCandidateList = [0, 2, -2, 4, -4]
                        for uvScaleIdx, uvScaleLogCandidate in enumerate(uvScaleLogCandidateList[:cfg.nUvScaleStep]):
                            uvScaleLogInit = th.tensor([uvScaleLogCandidate, uvScaleLogCandidate])
                            uvTransInit = th.tensor([0, 0])
                            for ang in range(angStep):
                                uvScaleLog = uvScaleLogInit
                                uvRot = th.tensor([angBase * ang])
                                uvTrans = uvTransInit
                                uvDict = {'rot': uvRot, 'trans': uvTrans, 'scaleBase': uvScaleBase, 'logscale': uvScaleLog}

                                sample_rendered_img = renderObj.eval(mData['basecolor'], mData['normal'], mData['roughness'], \
                                            uvDict=uvDict, eps=1e-10) # clamp to (eps, 1), ** (1/2.2)
                                sample_rendered_img = adjustColor(sample_rendered_img, pos_render, inputData['lossWeightMap'])
                                if cfg.graphSelectToGray:
                                    # Convert to grayscale
                                    sample_rendered_img = toGrayscale(sample_rendered_img)

                                err, err_stat, err_pix, err_vgg = \
                                    computeError(pos_render, sample_rendered_img, optObjectiveDict, renderObj)
                                errAll.append(err)
                        errAll = th.min(th.stack(errAll, dim=1), dim=1)[0] # Collect the lowest error!
                        errList.append(err)
                    errFull = th.cat(errList, dim=0)
                    graphLabelTensor = th.LongTensor(graphLabelList)
                    knn = errFull.topk(cfg.kNum, largest=False)
                    votes = graphLabelTensor[knn.indices]
                    hist = np.histogram(votes.cpu().numpy(), bins=np.arange(len(graphList)+1), weights=1/knn.values.cpu().numpy(), density=True)[0]
                    selectedGraph = graphList[np.argmax(hist)]

                    print('Selected %s graph!' % selectedGraph)
                    
                f.write('%s %s\n' % (matName, selectedGraph) )
    
    else:
        print('Please set usekNN to true to use kNN-based graph selection!')
        assert(False)

        
if __name__ == "__main__":
    print('\n')
    parser = argparse.ArgumentParser(description='kNN-based Graph Selection for PhotoScene')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    main(cfg)
