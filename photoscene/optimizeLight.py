import os
import argparse
import glob
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
import torch as th
import cv2
import json
import copy

from utils.io import loadBinary, load_cfg, readViewList, getMaterialSavePathDict, readModelStrIdDict, \
    readViewGraphDict, readNormal
from utils.xml import getObjInfoDict, getOptUvDict, uvApplyAndSaveObjects, updateXmlNewMatAndShapePath, getBsdfIdDict, \
    updateXML
from utils.render import renderPass


def checkWindow(matDict):
    for mat in matDict.keys():
        if 'window' in mat:
            return True
    return False


def readMatPartMask(filePath, partIdDict, if_resize=False, imWidth=None, imHeight=None):
    matMap = loadBinary(
        filePath, channels=3, dtype=np.int32, if_resize=if_resize, imWidth=imWidth, imHeight=imHeight).squeeze()
    cadIdMap = matMap[:, :, 0]
    matIdMap = matMap[:, :, 1]
    cadId, matId = partIdDict['cadId'], partIdDict['matId']
    cadMask = cadIdMap == cadId
    matMask = matIdMap == matId
    partMask = cadMask * matMask
    return partMask


def jointMask(orMatPart, orNormal, snNormal):
    normalMask = th.sum(orNormal * snNormal, dim=1) > 0.95
    return orMatPart * normalMask


def readImgRaw(imgPath, imH=None, imW=None):
    im = cv2.imread(imgPath, -1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if imH is not None and imW is not None:
        im = cv2.resize(im, ( imW, imH), interpolation=cv2.INTER_AREA)
    im = th.from_numpy(im).permute(2, 0, 1).unsqueeze(0)
    return im


def solveLinear(A, b):
    # A: m x n ; b: m
    AT = th.transpose(A, 0, 1)  # n x m
    return th.matmul(th.inverse(th.matmul(AT, A)), th.matmul(AT, b))  # n


def genCombinedLightXMLs(xmlSaveDir, lightList, coefList):
    noLightXml = osp.join(xmlSaveDir, 'main_noLight.xml')
    outXml     = osp.join(xmlSaveDir, 'main_combined.xml')

    # fetch previous intensity
    cNodeList = []  # collect ceiling light node
    envScale = 0.000001
    for lId, lName in enumerate(lightList):
        singleXml = osp.join(xmlSaveDir, 'main_%s.xml' % lName)
        tree = ET.parse(singleXml)
        root = tree.getroot()
        for child in root:
            if 'cLight' in lName and child.tag == 'shape' and 'ceilingLight' in child.attrib['id']:
                newNode = copy.deepcopy(child)
                for child2 in newNode:
                    if child2.tag == 'emitter':
                        oldRGB = child2[0].attrib['value'].split(' ')
                        newR = float(oldRGB[0]) * coefList[lId][0]
                        newG = float(oldRGB[1]) * coefList[lId][1]
                        newB = float(oldRGB[2]) * coefList[lId][2]
                        child2[0].set('value', '%f %f %f' % (newR, newG, newB))
                        cNodeList.append(newNode)
                        print('Update %s, New rgb scale: %f %f %f' % (lName, newR, newG, newB))
                        break
                break
            if 'areaLight' in lName and child.tag == 'shape' and 'areaLight' in child.attrib['id']:
                newNode = copy.deepcopy(child)
                for child2 in newNode:
                    if child2.tag == 'emitter':
                        oldRGB = child2[0].attrib['value'].split(' ')
                        newR = float(oldRGB[0]) * coefList[lId][0]
                        newG = float(oldRGB[1]) * coefList[lId][1]
                        newB = float(oldRGB[2]) * coefList[lId][2]
                        child2[0].set('value', '%f %f %f' % (newR, newG, newB))
                        cNodeList.append(newNode)
                        print('Update %s, New rgb scale: %f %f %f' % (lName, newR, newG, newB))
                        break
                break
            if 'eLight' in lName and child.tag == 'emitter':
                for child2 in child:
                    if child2.attrib['name'] == 'filename':
                        oldEnvName = child2.attrib['value']
                        print('Read old env map from %s' % oldEnvName)
                        envOri = cv2.imread(oldEnvName, -1)  # BGR
                        envR = envOri[:, :, 2] * coefList[lId][0].item()
                        envG = envOri[:, :, 1] * coefList[lId][1].item()
                        envB = envOri[:, :, 0] * coefList[lId][2].item()
                        newEnvName = osp.join(xmlSaveDir, 'main_combined.hdr')
                        print(
                            'Update %s, New env coefs: ' % lName, coefList[lId][0], coefList[lId][1], coefList[lId][2])
                        cv2.imwrite(newEnvName, np.stack([envB, envG, envR], axis=2))
                        print('New env map saved at %s' % newEnvName)
                        envScale = 1.0

                break

    newTree = ET.parse(noLightXml)
    newRoot = newTree.getroot()
    cnt = 0
    for child in newRoot:
        if child.tag == 'emitter':
            for child2 in child:
                if child2.attrib['name'] == 'scale':
                    child2.set('value', '%f' % envScale)
            break
        cnt += 1
    for cId, clNode in enumerate(cNodeList):
        newRoot.insert(cnt - 1 + cId, clNode)
    newTree.write(outXml)
    print('Write a new xml file for combined light: %s' % outXml)
    return outXml


def saveRawImg(imgPath, imgTensor):
    # imgTensor: 1 x 3 x H x W
    imgArr = imgTensor.squeeze().permute(1, 2, 0).numpy()[:, :, ::-1]  # RGB2BGR
    imgArr = ( (imgArr ** (1 / 2.2)).clip(0, 1) * 255).astype(np.uint8)
    if not osp.exists(osp.dirname(imgPath)):
        os.system('mkdir -p %s' % osp.dirname(imgPath))
    cv2.imwrite(imgPath, imgArr)


if __name__ == '__main__':
    # print('')
    parser = argparse.ArgumentParser(description='Script for lighting optimization')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    xmlBaseDir = cfg.xmlDir
    xmlBaseFile = cfg.xmlMedianFile
    xmlSaveDir = cfg.xmlSingleDir
    objSaveDir = cfg.meshSaveInitDir
    if not osp.exists(objSaveDir):
        os.system('mkdir -p %s' % objSaveDir)
    renderDir = cfg.renderSingleLightDir
    sampledCamFile = cfg.sampledCamFile
    sampledViewFile = cfg.sampledCamIdFile  # sampledViewList
    candidateCamIdList = readViewList(sampledViewFile)
    matSavePathDict = getMaterialSavePathDict(cfg, mode='init')  # partName: fileList

    # Generate new xml file, apply uv to save new obj, update material paths
    print('\n---> Preparing new single-light xml files and apply optimized material parameters to objects!\n')
    objInfoDict = getObjInfoDict(xmlBaseFile)
    # Collect material UV dicts and apply to OBJ
    objMatUVDictPath = osp.join(objSaveDir, 'objMatUVDict.json')
    if not osp.exists(objMatUVDictPath):
        # fetch objInfo from xml, fetch uv for each material, combine to one uv dict
        objMatUVDict = getOptUvDict(cfg, xmlBaseFile)
        with open(objMatUVDictPath, 'w') as fp:
            json.dump(objMatUVDict, fp, indent=4)
        objInfoDictNewUV = uvApplyAndSaveObjects(objInfoDict, objMatUVDict, objSaveDir, isSave=True)
    else:
        objMatUVDict = {}
        with open(objMatUVDictPath, 'r') as fp:
            objMatUVDictTemp = json.load(fp)
            for k, v in objMatUVDictTemp.items():
                objMatUVDict[int(k)] = v
        objInfoDictNewUV = uvApplyAndSaveObjects(objInfoDict, objMatUVDict, objSaveDir, isSave=False)

    objInfoDictNewUV2 = {}
    for objID, objInfo in objInfoDictNewUV.items():
        if not objInfo['isContainer'] and not objInfo['isEmitter']:
            objInfoDictNewUV2[objInfo['shape_id']] = objInfo

    # Update objDict to each light xml file and render one-light-at-a-time images #
    print('\n---> Rendering single-light images for lighting optimization...')
    xmlFileList = glob.glob(osp.join(cfg.xmlDir, '*Light*.xml'))
    for xmlFileLight in xmlFileList:
        objInfoDictLight = getObjInfoDict(xmlFileLight)
        for objID, objInfo in objInfoDictLight.items():
            shape_ID = objInfo['shape_id']
            if shape_ID in objInfoDictNewUV2.keys() and not objInfo['isContainer'] and not objInfo['isEmitter']:
                objInfoDictLight[objID] = objInfoDictNewUV2[shape_ID]

        xmlFileNew = osp.join(xmlSaveDir, osp.basename(xmlFileLight))
        xmlFileNew = updateXmlNewMatAndShapePath(xmlFileLight, xmlFileNew, matSavePathDict,
            objInfoDictLight, nSample=cfg.nSamplePerPixelSingle)

        lightName = osp.basename(xmlFileLight).replace('main_', '').replace('.xml', '')
        if lightName == 'noLight':
            continue

        renderPass(cfg, osp.join(renderDir, lightName), 'image', xmlFileNew, sampledCamFile,
            candidateCamIdList)

    # Compute light coeff. and exposure for each view #
    print('\n---> Computing light coeff. and exposure...')
    modelStrIdDict = readModelStrIdDict(cfg.modelIdFile)
    bsdfIdDict = getBsdfIdDict(cfg.xmlFile, modelStrIdDict)
    mats = readViewGraphDict(cfg.selectedViewFile, cfg.selectedGraphFile)
    haveWindow = checkWindow(mats)

    areaLightXmls = glob.glob(osp.join(xmlSaveDir, '*areaLight*'))
    areaList = [ osp.basename(fn).replace('main_', '').replace('.xml', '') for fn in areaLightXmls]
    # lightList = areaList
    lightList = areaList + ['eLight'] if haveWindow else areaList

    AListAll = []
    bList    = []
    for vId in candidateCamIdList:
        orNormal = readNormal(cfg.normalPathById % vId, refW=cfg.imgWidth, refH=cfg.imgHeight)
        snNormal = readNormal(cfg.invRenderNormalById % vId, refW=cfg.imgWidth, refH=cfg.imgHeight)
        mask = th.zeros(1, 1, cfg.imgHeight, cfg.imgWidth).bool()
        refImg = th.zeros(1, 3, cfg.imgHeight, cfg.imgWidth)
        for mm in mats.keys():  # collect mask for all part regions
            partIdDict = bsdfIdDict[mm]
            matPartMask = readMatPartMask(cfg.partIdPathById % vId, partIdDict, if_resize=True,
                imWidth=cfg.imgWidth, imHeight=cfg.imgHeight)
            matPartMask = th.from_numpy(matPartMask).unsqueeze(0).unsqueeze(1)
            matPartMaskJoint = jointMask(matPartMask, orNormal, snNormal)
            blendedWarpedRef = (th.from_numpy(cv2.cvtColor(cv2.resize(cv2.imread( cfg.photoOriById % vId, -1),
                (cfg.imgWidth, cfg.imgHeight)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0) / 255.0)

            mask = matPartMaskJoint * matPartMaskJoint + (~matPartMaskJoint) * mask
            refImg = matPartMaskJoint * blendedWarpedRef + (~matPartMaskJoint) * refImg

        AList = []
        for lName in lightList:
            imPath = cfg.imageSLPathByLightId % (lName, vId)
            img = readImgRaw(imPath, imW=cfg.imgWidth, imH=cfg.imgHeight)
            imgSelect = img.view(3, -1)[:, mask.view(-1)]  # 3 x nSelectPix
            AList.append(imgSelect)
        A = th.stack(AList, dim=2)  # 3 x nSelectPix x nLight
        AListAll.append(A)

        refImg = refImg ** 2.2
        b = refImg.view(3, -1)[:, mask.view(-1)]  # 3 x nSelectPix
        bList.append(b)
    AFull = th.cat(AListAll, dim=1)  # 3 x nSelectPixAllView x nLight
    bFull = th.cat(bList, dim=1)  # 3 x nSelectPixAllView

    # Remove light which doesn't contribute #
    maskInit = th.sum(AFull, dim=(0, 1)) > 1e-2  # nLight
    AFull = AFull[:, :, maskInit]  # 3 x nSelectPixAllView x nLight(>1e-10)
    lightListInit = lightList
    lightList = []
    for idx in range(maskInit.shape[0]):
        if maskInit[idx]:
            lightList.append(lightListInit[idx])

    check = th.ones(len(lightList)).bool()  # nLight
    checkFull = check
    cond = True
    nRun = 0
    while (~cond):
        nRun += 1
        coefR = solveLinear(AFull[0, :, checkFull], bFull[0, :])  # nLight
        coefG = solveLinear(AFull[1, :, checkFull], bFull[1, :])
        coefB = solveLinear(AFull[2, :, checkFull], bFull[2, :])
        coef  = th.stack([coefR, coefG, coefB], dim=1)  # nLight x 3
        check = (coefR > 0) * (coefG > 0) * (coefB > 0)
        idx = th.nonzero(checkFull, as_tuple=True)[0]
        checkFull[idx] = check
        cond = check.long().sum() == check.size(0)

    newLightList = []
    coefList = []
    for i in range(checkFull.size(0)):
        if checkFull[i]:
            newLightList.append(lightList[i])
    for i in range(check.size(0)):
        coefList.append(coef[i, :])

    xmlCombined = genCombinedLightXMLs(xmlSaveDir, newLightList, coefList)

    # Compute Exposure per view
    exposureDict = {}
    for idx, vId in enumerate(candidateCamIdList):
        orNormal = readNormal(cfg.normalPathById % vId, refW=cfg.imgWidth, refH=cfg.imgHeight)
        snNormal = readNormal(cfg.invRenderNormalById % vId, refW=cfg.imgWidth, refH=cfg.imgHeight)
        mask = th.zeros(1, 1, cfg.imgHeight, cfg.imgWidth).bool()
        refImg = th.zeros(1, 3, cfg.imgHeight, cfg.imgWidth)
        for mm in mats.keys():  # collect mask for all part regions
            partIdDict = bsdfIdDict[mm]
            matPartMask = readMatPartMask(cfg.partIdPathById % vId, partIdDict, if_resize=True,
                imWidth=cfg.imgWidth, imHeight=cfg.imgHeight)
            matPartMask = th.from_numpy(matPartMask).unsqueeze(0).unsqueeze(1)
            matPartMaskJoint = jointMask(matPartMask, orNormal, snNormal)
            blendedWarpedRef = (th.from_numpy(cv2.cvtColor(cv2.resize(cv2.imread( cfg.photoOriById % vId, -1),
                (cfg.imgWidth, cfg.imgHeight)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0) / 255.0)

            mask = matPartMaskJoint * matPartMaskJoint + (~matPartMaskJoint) * mask
            refImg = matPartMaskJoint * blendedWarpedRef + (~matPartMaskJoint) * refImg

        if th.sum(mask) > 0:
            AList = []
            for lName in newLightList:
                imPath = cfg.imageSLPathByLightId % (lName, vId)
                img = readImgRaw(imPath, imW=cfg.imgWidth, imH=cfg.imgHeight)
                imgSelect = img.view(3, -1)[:, mask.view(-1)]  # 3 x nSelectPix
                AList.append(imgSelect)
            A = th.stack(AList, dim=2)  # 3 x nSelectPix x nLight

            refImg = refImg ** 2.2
            b = refImg.view(3, -1)[:, mask.view(-1)]  # 3 x nSelectPix

            A2 = th.matmul(A, th.stack(coefList, dim=1).unsqueeze(-1)).squeeze(-1)
            # [3 x nSelectPix x nLight] * [3 x nLight x 1] = [3 x nSelectPix x 1] -> 3 x nSelectPix
            exposure = th.sum(b, dim=1) / th.sum(A2 + 1e-10, dim=1)  # 3-dim
        else:
            exposure = th.ones(3)  
        exposureDict[str(vId)] = exposure

    # Adjust Warped Target Images
    for mat in mats.keys():  # for each view
        vId = mats[mat]['vId']
        exposure = exposureDict[str(vId)]
        refWapredImg = (readImgRaw(cfg.targetGeoFromPhotoByNameId % (mat, vId),
            imW=cfg.imgWidth, imH=cfg.imgHeight) / 255.0) ** 2.2  # 1 x 3 x H x W
        saveRawImg(cfg.photoWarpedByNameId % (mat, vId), refWapredImg / exposure.view(1, 3, 1, 1))

    # Render per-pixel lighting for each view
    print('\n---> Rendering combined-light per-pixel lighting...')
    updateXML(xmlCombined, xmlFileNew=xmlCombined, matInfoDict=None, objInfoDict=None, envInfoDict=None,
                nSample=cfg.nSamplePerPixelLight, imWidth=cfg.imgWidth, imHeight=cfg.imgHeight)
    renderPass(cfg, cfg.renderCombinedLightDir, 'light', xmlCombined, sampledCamFile,
            candidateCamIdList)

    print('\n---> Lighting optimization is done!\n\n')
