import os
import argparse
import glob
import os.path as osp
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image

from utils.io import *
from utils.xml import *
import json
import math


def copyAndUpdateXmlDataPath(xmlFileOld, meshDir, matOriDir, envDir, xmlFileNew):
    tree = ET.parse(xmlFileOld)
    root = tree.getroot()
    for child in root: 
        if child.tag == 'shape':
            for child2 in child:
                if child2.tag == 'string' and child2.attrib['name'] == 'filename':
                    oldShapeName = child2.attrib['value']
                    newShapeName = oldShapeName.replace('/home/yyyeh/Datasets/OpenRoomRaw/mesh', meshDir)
                    child2.set('value', newShapeName)
        if child.tag == 'bsdf': # Modify is to avoid read path error
            for child2 in child:
                if child2.tag == 'texture':
                    oldMatPath = child2[0].attrib['value']
                    newMatPath = oldMatPath.replace('/home/yyyeh/Datasets/BRDFOriginDataset', matOriDir)
                    child2[0].set('value', newMatPath)
        if child.tag == 'emitter': # Modify is to avoid read path error
            for child2 in child:
                if child2.tag == 'string' and child2.attrib['name'] == 'filename':
                    hdrEnvName = xmlFileOld.replace('xml', 'hdr')
                    #print('Optimized env map is %s' % hdrEnvName)
                    if osp.exists(hdrEnvName):
                        newEnvName = hdrEnvName
                    else:
                        oldEnvName = child2.attrib['value']
                        newEnvName = oldEnvName.replace('/home/yyyeh/Datasets/OpenRoomRaw/envmap', envDir)
                    child2.set('value', newEnvName)
                    print('Set new envmap as %s' % newEnvName)

    tree.write(xmlFileNew)
    print('New Xml stored at %s' % xmlFileNew)

def saveDefaultMaterial(xmlFile, matSavePathDict, defaultMatSaveDir):
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    for child in root: 
        if child.tag == 'bsdf':
            bsdfStrId = child.attrib['id'] # material part name = cadcatID_objID_partID
            if bsdfStrId not in matSavePathDict.keys():
                #print('material part: %s not shown in this view, skip!' % bsdfStrId)
                continue

            defaultBsdfDir = osp.join(defaultMatSaveDir, bsdfStrId)
            if not os.path.exists(defaultBsdfDir):
                os.system('mkdir -p %s' % defaultBsdfDir )
            albedoFileOut = osp.join(defaultBsdfDir, 'basecolor.png')
            normalFileOut = osp.join(defaultBsdfDir, 'normal.png')
            roughFileOut = osp.join(defaultBsdfDir, 'roughness.png')

            isHomo = True
            bsdfPathDict = {}
            for child2 in child:
                if child2.tag == 'texture':
                    isHomo = False
                if child2.tag == 'texture' and child2.attrib['name'] == 'albedo':
                    bsdfPathDict['albedo'] = child2[0].attrib['value']
                if child2.tag == 'rgb' and child2.attrib['name'] == 'albedo':
                    bsdfPathDict['albedo'] = child2.attrib['value'] # float float float
                if child2.tag == 'rgb' and child2.attrib['name'] == 'albedoScale':
                    bsdfPathDict['albedoScale'] = child2.attrib['value'] # float float float

                if child2.tag == 'texture' and child2.attrib['name'] == 'normal':
                    bsdfPathDict['normal'] = child2[0].attrib['value']

                if child2.tag == 'texture' and child2.attrib['name'] == 'roughness':
                    bsdfPathDict['roughness'] = child2[0].attrib['value']
                if child2.tag == 'float' and child2.attrib['name'] == 'roughness':
                    bsdfPathDict['roughness'] = child2.attrib['value'] # float 
                if child2.tag == 'float' and child2.attrib['name'] == 'roughnessScale':
                    bsdfPathDict['roughnessScale'] = child2.attrib['value'] # float 

            rs, gs, bs = bsdfPathDict['albedoScale'].split(' ')
            roughs     = bsdfPathDict['roughnessScale']
            if isHomo:
                r, g, b = bsdfPathDict['albedo'].split(' ')
                rough   = bsdfPathDict['roughness']
                rescaled = lambda v, vs: int(float(v) * float(vs) * 255)
                albedo = Image.new('RGB', (512, 512), (rescaled(r, rs), rescaled(g, gs), rescaled(b, bs)))
                normal = Image.new('RGB', (512, 512), (128, 128, 255))
                normal = normal.save(normalFileOut)
                rough = Image.new('L', (512, 512), rescaled(rough, roughs))
            else:
                albedo = Image.open(bsdfPathDict['albedo']).convert('RGB')
                ar, ag, ab = albedo.split()
                ar = ar.point(lambda i: int( ( (i/255.0)**2.2 * float(rs) )**(1/2.2)*255 ) )
                ag = ag.point(lambda i: int( ( (i/255.0)**2.2 * float(gs) )**(1/2.2)*255 ) )
                ab = ab.point(lambda i: int( ( (i/255.0)**2.2 * float(bs) )**(1/2.2)*255 ) )
                albedo = Image.merge('RGB', (ar, ag, ab))
                os.system('cp %s %s' % (bsdfPathDict['normal'], normalFileOut))
                rough = Image.open(bsdfPathDict['roughness']).convert('L')
                rough = rough.point(lambda i: i * float(roughs))
            albedo = albedo.save(albedoFileOut)
            rough = rough.save(roughFileOut)

def updateXmlNewMatPath(xmlFileNew, matSavePathDict, optMatDir, isFast=False, useOriginalMaterial=False, isHQ=False, isRealScene=False):
    tree = ET.parse(xmlFileNew)
    root = tree.getroot()
    for child in root: 
        if child.tag == 'bsdf' and not useOriginalMaterial:
            bsdfStrId = child.attrib['id'] # material part name = cadcatID_objID_partID
            if bsdfStrId not in matSavePathDict.keys():
                print('material part: %s not shown in this view, skip!' % bsdfStrId)
                continue
            isHomo = True
            for child2 in child:
                if child2.tag == 'texture':
                    isHomo = False
                if child2.tag == 'texture' and child2.attrib['name'] == 'albedo':
                    fn = matSavePathDict[bsdfStrId][0]
                    child2[0].set('value', os.path.join(optMatDir, bsdfStrId, fn))

                if child2.tag == 'texture' and child2.attrib['name'] == 'normal':
                    fn = matSavePathDict[bsdfStrId][1]
                    child2[0].set('value', os.path.join(optMatDir, bsdfStrId, fn))

                if child2.tag == 'texture' and child2.attrib['name'] == 'roughness':
                    fn = matSavePathDict[bsdfStrId][2]
                    child2[0].set('value', os.path.join(optMatDir, bsdfStrId, fn))

                if child2.tag == 'rgb' and child2.attrib['name'] == 'albedoScale':
                    child2.set('value', '1.000 1.000 1.000')
                if child2.tag == 'float' and child2.attrib['name'] == 'roughnessScale':
                    child2.set('value', '1.000')
            if isHomo:    
                child2List = []
                for child2 in child:
                    child2List.append(child2)
                for child2 in child2List:
                    child.remove(child2)
                # uvScale
                if not isRealScene:
                    cond = 'scene' in bsdfStrId
                else:
                    cond = 'floor' in bsdfStrId.lower() or 'ceiling' in bsdfStrId.lower() or 'wall' in bsdfStrId.lower() or 'default' in bsdfStrId
                uvScale = '30.00' if cond else '1.000'
                m   = ET.SubElement(child, 'float', {'name': 'uvScale', 'value': uvScale})
                m.tail = '\n        '
                # albedo
                m   = ET.SubElement(child, 'texture', {'name': 'albedo', 'type': 'bitmap'})
                m.text = '\n            '
                fn = matSavePathDict[bsdfStrId][0]
                m2  = ET.SubElement(m,      'string', {'name': 'filename', 'value': os.path.join(optMatDir, bsdfStrId, fn) })
                m2.tail = '\n        '
                m.tail = '\n        '
                # albedoScale
                m   = ET.SubElement(child, 'rgb', {'name': 'albedoScale', 'value': '1.000 1.000 1.000'})
                m.tail = '\n        '
                # normal
                m   = ET.SubElement(child, 'texture', {'name': 'normal', 'type': 'bitmap'})
                m.text = '\n            '
                fn = matSavePathDict[bsdfStrId][1]
                m2  = ET.SubElement(m,      'string', {'name': 'filename', 'value': os.path.join(optMatDir, bsdfStrId, fn) })
                m2.tail = '\n        '
                m.tail = '\n        '
                # roughness
                m   = ET.SubElement(child, 'texture', {'name': 'roughness', 'type': 'bitmap'})
                m.text = '\n            '
                fn = matSavePathDict[bsdfStrId][2]
                m2  = ET.SubElement(m,      'string', {'name': 'filename', 'value': os.path.join(optMatDir, bsdfStrId, fn) })
                m2.tail = '\n        '
                m.tail = '\n        '
                # roughnessScale
                m   = ET.SubElement(child, 'float', {'name': 'roughnessScale', 'value': '1.000'})
                m.tail = '\n    '

        if child.tag == 'sensor':
            nSample = '16' if isFast else '1024' if isHQ else '128'
            for child2 in child:
                if child2.tag == 'sampler':
                    child2[0].set('value', nSample)
                elif child2.tag == 'film':
                    if isHQ:
                        child2[0].set('value', '640')
                        child2[1].set('value', '480')

    tree.write(xmlFileNew)
    print('New Xml stored at %s' % xmlFileNew)

# already in io.py
# def readViewGraphDict(viewDictDir):
#     matDict = {}
#     viewDictFile = osp.join(viewDictDir, 'selectedViewDict.txt') 
#     with open(viewDictFile, 'r') as f:
#         for line in f.readlines():
#             mat, vId = line.strip().split(' ')
#             if mat is not None and vId != '-1':
#                 matDict[mat] = {'vId': int(vId)}
#     graphDictFile = osp.join(viewDictDir, 'selectedGraphDict.txt') 
#     with open(graphDictFile, 'r') as f:
#         for line in f.readlines():
#             mat, graph = line.strip().split(' ')
#             if mat is not None and mat in matDict.keys():
#                 matDict[mat]['graph'] = graph
#     return matDict

def readViewList(sampledViewFile):
    with open(sampledViewFile, 'r') as f:
        candidateCamIdList = f.readline().strip().split(' ')
        candidateCamIdList = [int(cId) for cId in candidateCamIdList]
    return candidateCamIdList

def hdr2ldr(rgbeName, ldrName, noNormalize=False):
    im = cv2.imread(rgbeName, -1 )
    if not noNormalize:
        imMean =  np.mean(im )
        im = im / imMean * 0.5
    im = np.clip(im, 0, 1)
    im = im ** (1.0/2.2 )
    im = (255 * im).astype(np.uint8 )
    cv2.imwrite(ldrName, im )
    print('saving %s' % ldrName )

def renderPass(renderDir, passName, xmlFile, camFile, camIdList, isOverwrite=False, noNormalize=False, \
                program=None, programLight=None, programUV=None):

    renderDict = {'image': {'fn': 'im', 'ext': 'rgbe'}, 
                    'albedo': {'fn': 'imbaseColor', 'ext': 'png'}, 
                    'normal': {'fn': 'imnormal', 'ext': 'png'},
                    'rough': {'fn': 'imroughness', 'ext': 'png'},
                    'light': {'fn': 'imenv', 'ext': 'hdr'},
                    'uvCoord': {'fn': 'imuvcoord', 'ext': 'dat'}, }
    fn, ext = renderDict[passName]['fn'], renderDict[passName]['ext']

    outDir = osp.join(renderDir, passName)
    if not osp.exists(outDir):
        os.system('mkdir -p %s' % outDir)
    outFile = osp.join(outDir, 'im.hdr')

    outList = [osp.join(outDir, '%s_%d.%s' % (fn, int(vId), ext)) for vId in camIdList]
    if checkImgListExist(outList) and not isOverwrite:
        print('All images are rendered! Pass!')
    else:
        if passName == 'image':
            assert(program is not None)
            cmd = '%s -f %s -c %s -o %s -m 0 --maxIteration 4' % (program, xmlFile, camFile, outFile )
        elif passName == 'albedo':
            assert(program is not None)
            cmd = '%s -f %s -c %s -o %s -m 1' % (program, xmlFile, camFile, outFile )
        elif passName == 'normal':
            assert(program is not None)
            cmd = '%s -f %s -c %s -o %s -m 2' % (program, xmlFile, camFile, outFile )
        elif passName == 'rough':
            assert(program is not None)
            cmd = '%s -f %s -c %s -o %s -m 3' % (program, xmlFile, camFile, outFile )
        elif passName == 'light':
            assert(programLight is not None)
            cmd = '%s -f %s -c %s -o %s -m 7' % (programLight, xmlFile, camFile, outFile ) # default sampleNum is 25600
        elif passName == 'uvCoord':
            assert(programUV is not None)
            cmd = '%s -f %s -c %s -o %s -m 7' % (programUV, xmlFile, camFile, outFile )
        else:
            assert(False)

        print('Rendering %s in %s! ' % (passName, outDir))
        print(cmd)
        os.system(cmd)  

        with open(camFile, 'r') as f:
            for line in f.readlines():
                cNum = int(line.strip())
                break
        print('Total valid views: %d' % cNum)

        # Adjust View ID
        print('Adjusting view ID for %s ...' % outDir)
        tmpDir = outDir.replace(passName, '%s_tmp' % passName)
        os.system('mkdir -p %s' % tmpDir)
        for j in range(cNum):
            i = cNum - j - 1
            old = osp.join(outDir, '%s_%d.%s' % (renderDict[passName]['fn'], i+1, renderDict[passName]['ext'])  )
            assert(osp.exists(old))
            new = osp.join(tmpDir, '%s_%s.%s' % (renderDict[passName]['fn'], camIdList[i], renderDict[passName]['ext']) )
            cmd = 'mv %s %s' % (old, new )
            #print(cmd)
            os.system( cmd)
        assert(osp.exists(tmpDir))
        os.system('rm -r %s' % outDir)
        os.system('mv %s %s' % (tmpDir, outDir))
        print('Finish rendering locally!')

        if passName == 'image':
            # save png format
            for selectedId in camIdList:
                rgbeName = osp.join(outDir, 'im_%s.rgbe' % (selectedId))
                ldrName = rgbeName.replace('.rgbe', '.png')
                hdr2ldr(rgbeName, ldrName, noNormalize=noNormalize)
            print('Converted HDR to LDR!')

        if passName == 'uvCoord':
            # save visualization format
            for selectedId in camIdList:
                uvPath = osp.join(outDir, 'imuvcoord_%s.dat' % (selectedId))
                uvMap = loadBinary(uvPath, channels=2)[0, :] # 1 x refH x refW x 2 -> refH x refW x 2
                im = uvMap - np.floor(uvMap)
                im = Image.fromarray( (np.concatenate([im, np.ones((im.shape[0], im.shape[1], 1))], axis=2) * 255.0).astype(np.uint8) )
                im.save(os.path.join(uvPath.replace('.dat', '_vis.jpg')))

        if passName == 'light':
            for selectedId in camIdList:
                envFile = osp.join(outDir, 'imenv_%d.hdr' % (selectedId))
                envmap = cv2.imread(envFile, -1)
                if envmap.shape[0] == 120*8 and envmap.shape[1] == 160*16:
                    envHeight, envWidth = 8, 16
                elif envmap.shape[0] == 120*16 and envmap.shape[1] == 160*32:
                    envHeight, envWidth = 16, 32
                else:
                    assert(False)
                envmap = envmap.reshape(120, envHeight, 160, envWidth, 3)
                envmap = np.ascontiguousarray(envmap.transpose([4, 0, 2, 1, 3] ) ) # 3 x 120 x 160 x envHeight x envWidth

                envmap = np.transpose(envmap, [1, 2, 3, 4, 0] ) # 120 x 160 x envHeight x envWidth x 3
                nrows, ncols, gap= 24, 16, 1
                envRow, envCol = envmap.shape[0], envmap.shape[1]

                interY = int(envRow / nrows )
                interX = int(envCol / ncols )

                lnrows = len(np.arange(0, envRow, interY) )
                lncols = len(np.arange(0, envCol, interX) )

                lenvHeight = lnrows * (envHeight + gap) + gap
                lenvWidth = lncols * (envWidth + gap) + gap

                envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
                for r in range(0, envRow, interY ):
                    for c in range(0, envCol, interX ):
                        rId = int(r / interY )
                        cId = int(c / interX )

                        rs = rId * (envHeight + gap )
                        cs = cId * (envWidth + gap )
                        envmapLarge[rs : rs + envHeight, cs : cs + envWidth, :] = envmap[r, c, :, :, :]

                envmapLarge = np.clip(envmapLarge, 0, 1)
                envmapLarge = (255 * (envmapLarge ** (1.0/2.2) ) ).astype(np.uint8 )
                cv2.imwrite(envFile.replace('.hdr', '_vis.jpg'), envmapLarge[:, :, ::-1] )

def getMaterialSavePathDict(matOptDir, selectedGraphDict, modeName):
    matSavePathDict = {}
    for partName, selectedGraph in selectedGraphDict.items():
        fileList = []
        for matParam in ['basecolor', 'normal', 'roughness']:
            # fileName = os.path.join(matName, '%s.png' % matParam) 
            fileName = osp.join(partName, selectedGraph, modeName, matParam, '%s.png' % matParam)
            fullPath = os.path.join(matOptDir, fileName)
            if not os.path.exists(fullPath):
                print('Warning! %s not exist!' % fullPath)
                assert(False)
            fileList.append(fullPath)
        matSavePathDict[partName] = fileList
    return matSavePathDict

def getOptUvDict(xmlFile, matOptDir, modeName, selectedGraphDict, warpInfoDir, selectedViewDict):
    # Apply New UV paramters under matOptDir results to xmlFile
    # Return a objMatUVDict for objects in xmlFile
    #      {objID: {mat: uvDict}}
    # matOptDir: osp.join(orSnViewDir, expName, optMat)
    objMatUVDict = {}

    objInfoDict = getObjInfoDict(xmlFile)
    matInfoDict = getMatInfoDict(xmlFile)

    initUVScaleDict = {}
    for matId, matInfo in matInfoDict.items():
        initUVScaleDict[matInfo['bsdf_id']] = matInfo['uvScale']

    for objID in sorted(objInfoDict.keys()):
        if objInfoDict[objID]['isEmitter'] or objInfoDict[objID]['isContainer']:
            continue
        # objPath = objInfoDict[objID]['objPath']
        matList = objInfoDict[objID]['matList']
        objUVDict = {}
        for mat in matList:
            # initScale = initUVScaleDict[objID][mat]
            initScale = float(initUVScaleDict[mat])

            # # randomly generate uv scales, rotations, translations
            # uvDict = {}
            # scaleUV = initScale * (1.5 ** random.uniform(-3.5, 4.5)) if isRandomScale else initScale
            # uvDict['scaleU'] = scaleUV
            # uvDict['scaleV'] = scaleUV
            # uvDict['rot'] = random.uniform(-math.pi, math.pi) if isRandomRot else 0
            # uvDict['transU'] = random.uniform(-0.5, 0.5) if isRandomTrans else 0
            # uvDict['transV'] = random.uniform(-0.5, 0.5) if isRandomTrans else 0
            if mat in selectedGraphDict.keys():
                warpInfoFile = osp.join(warpInfoDir, mat, '%d_warpInfo.txt' % selectedViewDict[mat])
                assert(osp.exists(warpInfoFile))
                warpScale = 1
                with open(warpInfoFile, 'r') as f:
                    cnt = 0
                    for line in f.readlines():
                        print(line)
                        if cnt == 0:
                            cnt += 1
                            continue
                        else:
                            _, _, _, _, lOrX, lOrY, lSnX, lSnY = line.strip().split(' ')
                            xScale = float(lOrX) / float(lSnX)
                            yScale = float(lOrY) / float(lSnY)
                            warpScale = math.sqrt(xScale ** 2 + yScale ** 2) / math.sqrt(2)
                            # warpScale = max(float(lOrX) / float(lSnX), float(lOrY) / float(lSnY))
                            break
                
                graph = selectedGraphDict[mat]
                uvDictPath = osp.join(matOptDir, mat, graph, modeName, 'uvDict.json')
                with open(uvDictPath, 'r') as fp:
                    uvDict = json.load(fp)
                print(mat, warpScale)
            else:
                print('Part %s did not have optimized material. Use default identity UV transform.' % mat)
                uvDict = {'scaleU': 1, 'scaleV': 1, 'rot': 0, 'transU': 0, 'transV': 0}
                warpScale = 1
            uvDict['scaleU'] = uvDict['scaleU'] * initScale * warpScale
            uvDict['scaleV'] = uvDict['scaleV'] * initScale * warpScale
            objUVDict[mat] = uvDict
            
        objMatUVDict[objID] = objUVDict

    return objMatUVDict