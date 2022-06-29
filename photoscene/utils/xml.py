import xml.etree.ElementTree as ET
import os
import os.path as osp
from utils.io import *
from utils.render import *
import copy
from PIL import Image
import torch as th

def updateXML(xmlFileOld, xmlFileNew=None, matInfoDict=None, objInfoDict=None, envInfoDict=None, \
                nSample=None, imWidth=None, imHeight=None):
    # matSavePathDict specify the new material names and save directory
    # objSavePathDict: new OBJ paths 
    if matInfoDict is None:
        matSavePathDict = None
    else:
        matSavePathDict = {}
        for matID, matInfo in matInfoDict.items():
            matSavePathDict[matInfo['bsdf_id']] = matInfo

    if objInfoDict is None:
        objSavePathDict = None
    else:
        objSavePathDict = {}
        for objID, objInfo in objInfoDict.items():
            objSavePathDict[objInfo['shape_id']] = objInfo

    tree = ET.parse(xmlFileOld)
    root = tree.getroot()
    for child in root: 
        if child.tag == 'bsdf' and matInfoDict is not None: 
            bsdfStrId = child.attrib['id'] # material part name = cadcatID_objID_partID
            if bsdfStrId not in matSavePathDict.keys():
                print('material part: %s not shown in matInfoDict, skip updating!' % bsdfStrId)
                continue
            addBsdfContentNode(child,  matSavePathDict[bsdfStrId])

        if child.tag == 'shape' and objInfoDict is not None:
            shapeStrId = child.attrib['id'] # cadcatID_objID_object
            # objName = shapeStrId.replace('_object', '')
            if shapeStrId not in objSavePathDict.keys():
                print('shpae: %s not shown in objInfoDict, skip updating!' % shapeStrId)
                continue
            for child2 in child:
                if child2.tag == 'string' and child2.attrib['name'] == 'filename':
                    objPath = objSavePathDict[shapeStrId]['objPath']
                    if not osp.exists(objPath):
                        print('ObjFile %s not exist!' % objPath)
                        assert(False)
                    child2.set('value', objPath)
                if child2.tag == 'emitter' and objSavePathDict[shapeStrId]['isEmitter']:
                    child2[0].set('value', objSavePathDict[shapeStrId]['emitterVal'])

        if child.tag == 'emitter' and envInfoDict is not None: # Modify is to avoid read path error
            for child2 in child:
                if child2.tag == 'string' and child2.attrib['name'] == 'filename':
                    child2.set('value', envInfoDict['envPath'])

        if child.tag == 'sensor':
            for child2 in child:
                if child2.tag == 'sampler' and nSample is not None:
                    child2[0].set('value', str(nSample) )
                    #child2.set('type', 'independent')
                if child2.tag == 'film':
                    for child3 in child2:
                        if child3.attrib['name'] == 'width' and imWidth is not None:
                            child3.set('value', str(imWidth))
                        elif child3.attrib['name'] == 'height' and imHeight is not None:
                            child3.set('value', str(imHeight))
    if xmlFileNew is None:
        xmlFileNew = xmlFileOld.replace('.xml', '_new.xml')
    tree.write(xmlFileNew)
    print('New Xml stored at %s' % xmlFileNew)
    return xmlFileNew

def updateXMLFromRaw(xmlFileOld, meshDir, orSnViewDir, matOriDir, envDir, imHeight, imWidth, xmlFileNew=None, saveEmpty=False):
    # Modify paths and Save new XML file from the OpenRoomRaw XML files
    xmlInfoDict = getXmlInfoDict(xmlFileOld)

    objInfoDict = xmlInfoDict['shape']
    for objID, objInfo in objInfoDict.items():
        newPath = osp.join(meshDir, objInfo['objPath'].replace('../../../../../', ''))
        objInfoDict[objID]['objPath'] = newPath

    matInfoDict = xmlInfoDict['bsdf']
    for matID, matInfo in matInfoDict.items():
        if not matInfo['isHomo']:
            for matParam in ['albedo', 'normal', 'roughness']:
                newPath = osp.join(matOriDir, matInfo[matParam].replace('../../../../../BRDFOriginDataset/', '') )
                matInfoDict[matID][matParam] = newPath
        uvScale = '0.300' if 'scene' in matInfo['bsdf_id'] else '1.000'
        matInfoDict[matID]['uvScale'] = uvScale

    sensorInfoDict = xmlInfoDict['sensor']
    sensorInfoDict['width'] = str(imWidth)
    sensorInfoDict['height'] = str(imHeight)

    envInfoDict = xmlInfoDict['emitter']
    envInfoDict['envPath'] = osp.join(envDir, envInfoDict['envPath'].replace('../../../../../', '') )

    # No need to do so, above assigned by reference
    # xmlInfoDict['shape'] = objInfoDict
    # xmlInfoDict['bsdf'] = matInfoDict
    # xmlInfoDict['sensor'] = sensorInfoDict
    # xmlInfoDict['emitter'] = envInfoDict

    if xmlFileNew is None:
        xmlFileNew = osp.join(orSnViewDir, 'xml', 'main.xml')
    if not osp.exists(orSnViewDir):
        os.system('mkdir -p %s' % orSnViewDir)
    # xmlFileNew = updateXML(xmlFileOld, xmlFileNew=xmlFileNew, matInfoDict=matInfoDict, objInfoDict=objInfoDict, \
    #             envInfoDict=envInfoDict, nSample=None, imWidth=imWidth, imHeight=imHeight)
    xmlFileNew = createXmlFromInfoDict(xmlFileNew, xmlInfoDict)
    # print('New Xml stored at %s' % xmlFileNew)

    if saveEmpty:
        treeEmpty = ET.parse(xmlFileOld)
        rootEmpty = treeEmpty.getroot()
        nElementOld = len(rootEmpty)
        def removeChild(rootEmpty):
            for child in rootEmpty:
                if child.tag == 'bsdf' or child.tag == 'shape':
                    childId = child.attrib['id']
                    if 'scene' in childId or 'door' in childId or 'window' in childId:
                        continue
                else: # (child.tag == 'sensor' or child.tag == 'emitter')
                    continue
                rootEmpty.remove(child)
        removeChild(rootEmpty)
        nElementNew = len(rootEmpty)
        while nElementNew < nElementOld:
            nElementOld = nElementNew
            removeChild(rootEmpty)
            nElementNew = len(rootEmpty)
        treeEmpty.write(xmlFileNew.replace('.xml', '_empty.xml'))

    return xmlFileNew

def saveNewWhiteXML(xmlFile, imWidth=None, imHeight=None, outPath=None):
    whiteAlbedo = '0.900 0.900 0.900'
    whiteAlbedoScale = '1.000 1.000 1.000'
    whiteRough  = '0.700'
    whiteRoughScale = '1.000'

    matInfoDict = getMatInfoDict(xmlFile)
    for matID, matInfo in matInfoDict.items():
        newMatInfo = {'bsdf_id': matInfo['bsdf_id'], 'uvScale': matInfo['uvScale'], 'isHomo': True, \
                        'albedo': whiteAlbedo, 'albedoScale': whiteAlbedoScale, \
                        'roughness': whiteRough, 'roughnessScale': whiteRoughScale}
        matInfoDict[matID] = newMatInfo

    if outPath is None:
        savePath = xmlFile.replace('.xml', '_white.xml')
    else:
        savePath = outPath
    xmlFileNew = updateXML(xmlFile, xmlFileNew=savePath, matInfoDict=matInfoDict, objInfoDict=None, \
                envInfoDict=None, nSample=None, imWidth=imWidth, imHeight=imHeight)
    # print('New Xml stored at %s' % xmlFileNew)
    return xmlFileNew

def genBaselineMatAndPathDict(cfg, baseline, tag=None, refW=320, refH=240):
    # tag is used for first and second round materials
    # selectedViewFile = osp.join(orSnViewDir, 'selectedViewFullDict.txt')
    selectedViewFile = cfg.selectedFullDictFile
    selectedViewDict = readViewDict(selectedViewFile)
    baselineMatDict = {}
    if baseline == 'median' or baseline == 'invrender' or baseline == 'white':
        for mat, vId in selectedViewDict.items():
            # invRenderData = loadInvRenderData(orSnViewDir, vId, refH=refH, refW=refW)
            invRenderData = loadInvRenderData(cfg, vId, refH=refH, refW=refW)
            # maskSNPath = os.path.join(orSnViewDir, 'output', mat, '%d_maskSN.png' % vId)
            maskSNPath = cfg.maskPhotoByNameId % (mat, vId)
            mask_ref = th.from_numpy(np.asarray(Image.open(maskSNPath).resize((refW, refH))).astype(np.float)/255).unsqueeze(0).unsqueeze(1)
            mask_ref = mask_ref > 0.9
            if baseline == 'median':
                target_ref = invRenderData['im']
                median, medianIdx = th.median(target_ref.view(3, -1)[:, mask_ref.view(-1)], dim=1) # 3-d
                albColor = (int(median[0]*255), int(median[1]*255), int(median[2]*255))
                rouColor = (179)
                # Create median material
                smallMatDir = cfg.matMedianDirByName % mat
                # smallMatDir = osp.join(orSnViewDir, 'baselineMat', 'median', mat)
            elif baseline == 'invrender':
                alb_ref   = invRenderData['albedo']
                rough_ref = invRenderData['rough']
                alb, _ = th.median(alb_ref.view(3, -1)[:, mask_ref.view(-1)] ** (1/2.2), dim=1) # 3-d
                rough = th.median(rough_ref.view(-1)[mask_ref.view(-1)]) # 1-d
                albColor = (int(alb[0]*255), int(alb[1]*255), int(alb[2]*255))
                rouColor = (int(rough.item()*255))
                smallMatDir = cfg.matInvRenderDirByName % mat
                # smallMatDir = osp.join(orSnViewDir, 'baselineMat', 'invrender', mat)
            elif baseline == 'white':
                albColor = (230, 230, 230)
                rouColor = (179)
                smallMatDir = cfg.matWhiteDirByName % mat
                # smallMatDir = osp.join(orSnViewDir, 'baselineMat', 'white')
            else:
                assert(False)

            if not osp.exists(smallMatDir):
                os.system('mkdir -p %s' % smallMatDir)
            albedoPath = osp.join(smallMatDir, 'basecolor.png')
            Image.new(mode="RGB", size=(256,256), color=albColor).save(albedoPath)
            normalPath = osp.join(smallMatDir, 'normal.png')
            Image.new(mode="RGB", size=(256,256), color=(128, 128, 255)).save(normalPath)
            roughPath = osp.join(smallMatDir, 'roughness.png')
            Image.new(mode="L", size=(256,256), color=rouColor).save(roughPath)
            baselineMatDict[mat] = {'albedoPath': albedoPath, 'normalPath': normalPath, 'roughPath': roughPath}

    elif baseline == 'first' or baseline == 'second':
        assert(tag is not None)
        # optMatDir = osp.join(orSnViewDir, 'optMat') if baseline == 'first' else osp.join(orSnViewDir, 'optMatGL')
        # print('-------->>>>> %s %s' % (baseline, optMatDir))
        matOptViewDict = readViewGraphDict(cfg.selectedViewFile, cfg.selectedGraphFile)
        baselineMatDict = genSavePathDict(cfg, matOptViewDict, baseline, tag)

    else:
        assert(False)

    return baselineMatDict

def saveNewBaselineXML(cfg, baseline):
    assert(baseline in ['median', 'invrender', 'cls'])
    xmlFile = cfg.xmlWhiteFile
    outPath = cfg.xmlMedianFile if baseline is 'median' else cfg.xmlInvRenderFile if baseline is 'invrender' else None
    nSample = cfg.nSamplePerPixel
    imWidth, imHeight = cfg.imgWidth, cfg.imgHeight
    baselineMatDict = genBaselineMatAndPathDict(cfg, baseline=baseline, refW=imWidth, refH=imHeight)

    matInfoDict = getMatInfoDict(xmlFile)
    for matID, matInfo in matInfoDict.items():
        bsdfName = matInfo['bsdf_id']
        if bsdfName not in baselineMatDict.keys():
            # if there is no mask for this material
            continue
        newMatInfo = {'bsdf_id': matInfo['bsdf_id'], 'uvScale': matInfo['uvScale'], 'isHomo': False, \
                        'albedo': baselineMatDict[bsdfName]['albedoPath'], 'albedoScale': '1.000 1.000 1.000', \
                        'normal': baselineMatDict[bsdfName]['normalPath'], \
                        'roughness': baselineMatDict[bsdfName]['roughPath'], 'roughnessScale': '1.000'}
        matInfoDict[matID] = newMatInfo
    
    if outPath is None:
        savePath = xmlFile.replace('.xml', '_%s.xml' % baseline)
    else:
        savePath = outPath
    xmlFileNew = updateXML(xmlFile, xmlFileNew=savePath, matInfoDict=matInfoDict, objInfoDict=None, \
                envInfoDict=None, nSample=nSample, imWidth=imWidth, imHeight=imHeight)
    # print('New Xml stored at %s' % xmlFileNew)
    return xmlFileNew

def getObjInfoDict(xmlFile):
    objCnt = 0
    objInfoDict = {}
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    for child in root: 
        if child.tag == 'shape':
            shapeStrId = child.attrib['id'] # cadcatID_objID_object
            isEmitter = False
            emitterVal = None
            for child2 in child:
                if child2.tag == 'string' and child2.attrib['name'] == 'filename':
                    objPath = child2.attrib['value']
                if child2.tag == 'emitter':
                    isEmitter = True
                    for child3 in child2:
                        if child3.tag == 'rgb':
                            emitterVal = child3.attrib['value']

            cond1 = 'aligned_light.obj' in objPath
            cond2 = 'container.obj' in objPath
            # cond3 = objName in objInfoDict.keys()
            # if cond1 or cond2:
            #     continue
            isAlignedLight = True if cond1 else False
            isContainer = True if cond2 else False
            # if not osp.exists(objPath):
            #     print('ObjFile %s not exist!' % objPath)
            #     assert(False)

            matList = []
            for child2 in child:
                if child2.tag == 'ref' and child2.attrib['name'] == 'bsdf':
                    matList.append(child2.attrib['id'])
                if child2.tag == 'transform':
                    # Collect all transformations
                    opList = []
                    attribList = []
                    for child3 in child2:
                        op = child3.tag
                        x = child3.attrib['x']
                        y = child3.attrib['y']
                        z = child3.attrib['z']
                        if op == 'scale' or op == 'translate':
                            attrib = {'x': x, 'y': y, 'z': z}
                        elif op == 'rotate':
                            angle = child3.attrib['angle']
                            attrib = {'angle': angle, 'x': x, 'y': y, 'z': z}
                        else:
                            print('Invalid transform operation!')
                            assert(False)
                        opList.append(op)
                        attribList.append(attrib)
                    # Compute total scale
                    scaleList = []
                    for child3 in child2:
                        if child3.tag == 'scale':
                            x, y, z = child3.attrib['x'], child3.attrib['y'], child3.attrib['z']
                            scale = np.array([float(x), float(y), float(z)])
                            scaleList.append(scale)
                    scaleTotal = 1
                    for scale in scaleList:
                        scaleTotal = scaleTotal * scale
            # objInfoDict[objName] = {'path': objPath, 'parts': matList, 'scale': scale}
            objInfoDict[objCnt] = \
                {'shape_id': shapeStrId, 'objPath': objPath, 'matList': matList, 'scale': scaleTotal, \
                    'isEmitter': isEmitter, 'emitterVal': emitterVal, \
                    'transform_opList': opList, 'transform_attribList': attribList, \
                    'isAlignedLight': isAlignedLight, 'isContainer': isContainer}
            objCnt += 1
    return objInfoDict

def getMatInfoDict(xmlFile):
    # { 0: {'isHomo': ?, 'bsdf_id': ?, 'albedo': ?}, 1: {} }
    matCnt = 0
    matInfoDict = {}
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    for child in root: 
        if child.tag == 'bsdf':
            bsdfStrId = child.attrib['id'] # material part name = cadcatID_objID_partID
            bsdfDict = {'bsdf_id': bsdfStrId}
            isHomo = True
            for child2 in child:
                if child2.tag == 'texture':
                    isHomo = False
                if child2.tag == 'texture' and child2.attrib['name'] == 'albedo':
                    bsdfDict['albedo'] = child2[0].attrib['value']
                if child2.tag == 'texture' and child2.attrib['name'] == 'normal':
                    bsdfDict['normal'] =  child2[0].attrib['value']
                if child2.tag == 'texture' and child2.attrib['name'] == 'roughness':
                    bsdfDict['roughness'] = child2[0].attrib['value']
                if child2.tag == 'rgb' and child2.attrib['name'] == 'albedoScale':
                    bsdfDict['albedoScale'] = child2.attrib['value']
                if child2.tag == 'float' and child2.attrib['name'] == 'roughnessScale':
                    bsdfDict['roughnessScale'] = child2.attrib['value']
                if child2.tag == 'float' and child2.attrib['name'] == 'uvScale':
                    bsdfDict['uvScale'] = child2.attrib['value']
                
            if isHomo:
                for child2 in child:
                    if child2.tag == 'rgb' and child2.attrib['name'] == 'albedo':
                        bsdfDict['albedo'] = child2.attrib['value']
                    if child2.tag == 'rgb' and child2.attrib['name'] == 'albedoScale':
                        bsdfDict['albedoScale'] = child2.attrib['value']
                    if child2.tag == 'float' and child2.attrib['name'] == 'roughness':
                        bsdfDict['roughness'] = child2.attrib['value']
                    if child2.tag == 'float' and child2.attrib['name'] == 'roughnessScale':
                        bsdfDict['roughnessScale'] = child2.attrib['value']
                    if child2.tag == 'float' and child2.attrib['name'] == 'uvScale':
                        bsdfDict['uvScale'] = child2.attrib['value']

            bsdfDict['isHomo'] = isHomo
            matInfoDict[matCnt] = bsdfDict
            matCnt += 1
    return matInfoDict

def getSensorInfoDict(xmlFile):
    sensorInfoDict = {}
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    for child in root: 
        if child.tag == 'sensor': # Modify is to avoid read path error
            for child2 in child:
                if child2.tag == 'float' and child2.attrib['name'] == 'fov':
                    sensorInfoDict['fov'] = child2.attrib['value']
                if child2.tag == 'string' and child2.attrib['name'] == 'fovAxis':
                    sensorInfoDict['fovAxis'] = child2.attrib['value']
                if child2.tag == 'film' and child2.attrib['type'] == 'hdrfilm':
                    for child3 in child2:
                        if child3.tag == 'integer' and child3.attrib['name'] == 'width':
                            sensorInfoDict['width'] = child3.attrib['value']
                        if child3.tag == 'integer' and child3.attrib['name'] == 'height':
                            sensorInfoDict['height'] = child3.attrib['value']
                if child2.tag == 'sampler':
                    sensorInfoDict['sampler_type'] = child2.attrib['type']
                    for child3 in child2:
                        if child3.attrib['name'] == 'sampleCount':
                            sensorInfoDict['sample_count'] = child3.attrib['value']
    return sensorInfoDict

def getEnvInfoDict(xmlFile):
    envInfoDict = {}
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    for child in root: 
        if child.tag == 'emitter': # Modify is to avoid read path error
            for child2 in child:
                if child2.tag == 'string' and child2.attrib['name'] == 'filename':
                    envInfoDict['envPath'] = child2.attrib['value']
                if child2.tag == 'float' and child2.attrib['name'] == 'scale':
                    envInfoDict['scale'] = child2.attrib['value']
    return envInfoDict

def getXmlInfoDict(xmlFile):
    objInfoDict = getObjInfoDict(xmlFile)
    matInfoDict = getMatInfoDict(xmlFile)
    sensorInfoDict = getSensorInfoDict(xmlFile)
    envInfoDict = getEnvInfoDict(xmlFile)
    return {'shape': objInfoDict, 'bsdf': matInfoDict, 'sensor': sensorInfoDict, 'emitter': envInfoDict}

def saveObjWithUvAdjusted(objPath, matUVDict, saveRoot, matNameMapping=None):
    # matUVDict: specify UV params for each material in original OBJ file
    # matNameMapping: if using new material names, map new to old

    v, vn, vt, matFacesListDict, matList = getObjMatVertFaceDict(objPath)

    ### Adjust texture vertex ###
    vtNew = vt
    for mat in matList:
        vtCollect = []
        for face in matFacesListDict[mat]:
            vt1, vt2, vt3 = face['vt1'], face['vt2'], face['vt3']
            vtCollect += list([vt1, vt2, vt3])
            # vtCollect = list(set(vtCollect) | set([vt1, vt2, vt3]))
        vtCollect = list(set(vtCollect))
        if mat not in matUVDict.keys(): # no bsdf specified in xml file but there is material in obj file
            print('Warning: Attempt to adjust uv with material %s not specified the uv transformation! ' % mat)
            continue
        else:
            uvDict = matUVDict[mat]
        uvScale = np.array([uvDict['scaleU'], uvDict['scaleV']])
        c, s = np.cos(uvDict['rot']), np.sin(uvDict['rot'])
        uvRot = np.array(((c, -s), (s, c)))
        uvTrans = np.array([uvDict['transU'], uvDict['transV']])
        for vv in vtCollect:
            uvCoord = vt[vv-1]
            vtNew[vv-1] = np.matmul(uvRot, uvCoord * uvScale) + uvTrans

    if matNameMapping is not None:
        newMatFacesListDict = {}
        newMatList = sorted(matNameMapping.keys())
        for new, old in matNameMapping.items():
            newMatFacesListDict[new] = matFacesListDict[old]
    else:
        newMatList = matList
        newMatFacesListDict = matFacesListDict

    ### Save new mesh ###
    saveRoot = saveObjFromMatVertFaceDict(v, vn, vtNew, newMatFacesListDict, newMatList, saveRoot)

def uvApplyAndSaveObjects(objInfoDict, objMatUVDict, objSaveDir, isSave=True):
    for objID, objInfo in objInfoDict.items():
        objName = objInfo['shape_id'].replace('_object', '')
        objPath = objInfo['objPath']
        hasVt = checkVtExist(objPath)
        saveRoot = osp.join(objSaveDir, '%s.obj' % objName)
        if objID not in objMatUVDict.keys():
            print('Warning! %s not exist in uvDict! Skip converting UV for this object!' % objInfo['shape_id'])
        elif not hasVt:
            print('Warning! %s does not have Vertex Coord. ! Skip converting UV for this object!' % objInfo['shape_id'])
        else:
            if isSave:
                saveObjWithUvAdjusted(objPath, objMatUVDict[objID], saveRoot)
            objInfoDict[objID]['objPath'] = saveRoot
    return objInfoDict

def getInitUVScaleDict(xmlFile, baseScale=1, isPrint=False):
    initUVScaleDict = {}
    objInfoDict = getObjInfoDict(xmlFile)
    print('Computing initial UV Scale for %s ...' % xmlFile)
    for objID in sorted(objInfoDict.keys()):
        objName = objInfoDict[objID]['shape_id']
        transform_scale = objInfoDict[objID]['scale']
        objPath = objInfoDict[objID]['objPath']
        # print(objPath)
        # check if vt exists
        hasVt = checkVtExist(objPath)
        if hasVt:
            v, vn, vt, mat_faces, matList = getObjMatVertFaceDict(objPath)
            # objInfoDict[objID]['matList'] = []
            matScaleDict = {}
            for mat in matList:
                # objInfoDict[objID]['matList'].append(mat)
                vCollect = []
                vtCollect = []
                for face in mat_faces[mat]:
                    v1, v2, v3 = face['v1'], face['v2'], face['v3']
                    vt1, vt2, vt3 = face['vt1'], face['vt2'], face['vt3']
                    vCollect += list([v1, v2, v3])
                    vtCollect += list([vt1, vt2, vt3])
                    # vtCollect = list(set(vtCollect) | set([vt1, vt2, vt3]))
                vCollect = list(set(vCollect))
                vtCollect = list(set(vtCollect))

                vertLocList = []
                for vv in vCollect:
                    vertLocList.append(v[vv-1])
                vertLoc = np.stack(vertLocList, axis=0) * transform_scale[np.newaxis, :]
                # Compute the farthest distance from the center
                center = 0.5 * (np.amax(vertLoc, axis=0) + np.amin(vertLoc, axis=0))
                dist = np.amax(np.sqrt(np.sum((vertLoc - center[np.newaxis, :]) ** 2, axis=1)))

                texVertLocList = []
                for vv in vtCollect:
                    texVertLocList.append(vt[vv-1])
                texVertLoc = np.stack(texVertLocList, axis=0)
                center = 0.5 * (np.amax(texVertLoc, axis=0) + np.amin(texVertLoc, axis=0))
                distUV = np.amax(np.sqrt(np.sum((texVertLoc - center[np.newaxis, :]) ** 2, axis=1)))

                matScaleDict[mat] = dist / distUV * baseScale

            initUVScaleDict[objID] = matScaleDict
            if isPrint:
                print('[%s]' % (objName ), end='' )
                for k, v in matScaleDict.items():
                    print(' %s:%f' % (k, v), end='')
                print('')

    return initUVScaleDict

# https://stackoverflow.com/a/33956544
def indent(elem, level=0):
    i = "\n" + level*"    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def addBsdfContentNode(child, matInfo, addTail=True):
    isHomo = matInfo['isHomo']
    child2List = []
    for child2 in child:
        child2List.append(child2)
    for child2 in child2List:
        child.remove(child2)
    if not isHomo:
        # uvScale
        m   = ET.SubElement(child, 'float', {'name': 'uvScale', 'value': matInfo['uvScale']})
        m.tail = '\n        ' if addTail else ''
        # albedo
        m   = ET.SubElement(child, 'texture', {'name': 'albedo', 'type': 'bitmap'})
        m.text = '\n            ' if addTail else ''
        m2  = ET.SubElement(m,      'string', {'name': 'filename', 'value': matInfo['albedo'] })
        m2.tail = '\n        ' if addTail else ''
        m.tail = '\n        ' if addTail else ''
        # albedoScale
        m   = ET.SubElement(child, 'rgb', {'name': 'albedoScale', 'value': matInfo['albedoScale']})
        m.tail = '\n        ' if addTail else ''
        # normal
        m   = ET.SubElement(child, 'texture', {'name': 'normal', 'type': 'bitmap'})
        m.text = '\n            ' if addTail else ''
        m2  = ET.SubElement(m,      'string', {'name': 'filename', 'value': matInfo['normal'] })
        m2.tail = '\n        ' if addTail else ''
        m.tail = '\n        ' if addTail else ''
        # roughness
        m   = ET.SubElement(child, 'texture', {'name': 'roughness', 'type': 'bitmap'})
        m.text = '\n            ' if addTail else ''
        m2  = ET.SubElement(m,      'string', {'name': 'filename', 'value': matInfo['roughness'] })
        m2.tail = '\n        ' if addTail else ''
        m.tail = '\n        ' if addTail else ''
        # roughnessScale
        m   = ET.SubElement(child, 'float', {'name': 'roughnessScale', 'value': matInfo['roughnessScale']})
        m.tail = '\n    ' if addTail else ''
    else:
        m   = ET.SubElement(child, 'float', {'name': 'uvScale', 'value': matInfo['uvScale']})
        m.tail = '\n        ' if addTail else ''
        m   = ET.SubElement(child, 'rgb', {'name': 'albedo', 'value': matInfo['albedo']})
        m.tail = '\n        ' if addTail else ''
        m   = ET.SubElement(child, 'rgb', {'name': 'albedoScale', 'value': matInfo['albedoScale']})
        m.tail = '\n        ' if addTail else ''
        m   = ET.SubElement(child, 'float', {'name': 'roughness', 'value': matInfo['roughness']})
        m.tail = '\n        ' if addTail else ''
        m   = ET.SubElement(child, 'float', {'name': 'roughnessScale', 'value': matInfo['roughnessScale']})
        m.tail = '\n    ' if addTail else ''

def addShapeContentNode(child, objInfo, addTail=True):
    child2List = []
    for child2 in child:
        child2List.append(child2)
    for child2 in child2List:
        child.remove(child2)
    # Filename
    m  = ET.SubElement(child, 'string', {'name': 'filename', 'value': objInfo['objPath']})
    m.tail = '\n        ' if addTail else ''
    # Emitter
    if objInfo['isEmitter']:
        m  = ET.SubElement(child, 'emitter', {'type': 'area'})
        m.text = '\n            ' if addTail else ''
        m2  = ET.SubElement(m,      'rgb', {'value': objInfo['emitterVal'] })
        m2.tail = '\n        ' if addTail else ''
        m.tail = '\n        ' if addTail else ''
    # Transform
    m  = ET.SubElement(child, 'transform', {'name': 'toWorld'})
    m.text = '\n            ' if addTail else ''
    for idx, (op, attrib) in enumerate(zip(objInfo['transform_opList'], objInfo['transform_attribList'])):
        m2 = ET.SubElement(m,      op, attrib)
        tailIndent = '\n            ' if idx < len(objInfo['transform_opList']) - 1 else '\n        '
        m2.tail = tailIndent if addTail else ''
    tailIndent = '\n        ' if len(objInfo['matList']) > 0 else '\n    '
    m.tail = tailIndent if addTail else ''
    # BRDF
    if not objInfo['isEmitter']:
        for idx, mat in enumerate(objInfo['matList']):
            m  = ET.SubElement(child, 'ref', {'id': mat, 'name': 'bsdf'})
            tailIndent = '\n    ' if idx < len(objInfo['matList']) -1 else '\n'
            m.tail = tailIndent if addTail else ''

def addSensorContentNode(child, sensorInfo, addTail=True):
    child2List = []
    for child2 in child:
        child2List.append(child2)
    for child2 in child2List:
        child.remove(child2)
    # fov
    m  = ET.SubElement(child, 'float', {'name': 'fov', 'value': sensorInfo['fov']})
    m.tail = '\n        ' if addTail else ''
    # fovAxis
    m  = ET.SubElement(child, 'string', {'name': 'fovAxis', 'value': sensorInfo['fovAxis']})
    m.tail = '\n        ' if addTail else ''
    # film
    m  = ET.SubElement(child, 'film', {'type': 'hdrfilm'})
    m.text = '\n            ' if addTail else ''
    m2  = ET.SubElement(m,      'integer', {'name': 'width', 'value': sensorInfo['width'] })
    m2.tail = '\n            ' if addTail else ''
    m2  = ET.SubElement(m,      'integer', {'name': 'height', 'value': sensorInfo['height'] })
    m2.tail = '\n        ' if addTail else ''
    m.tail = '\n        ' if addTail else ''
    # sampler
    m  = ET.SubElement(child, 'sampler', {'type': sensorInfo['sampler_type']})
    m.text = '\n            ' if addTail else ''
    m2  = ET.SubElement(m,      'integer', {'name': 'sampleCount', 'value': sensorInfo['sample_count'] })
    m2.tail = '\n        ' if addTail else ''
    m.tail = '\n        ' if addTail else ''

def addEmitterContentNode(child, envInfo, addTail=True):
    child2List = []
    for child2 in child:
        child2List.append(child2)
    for child2 in child2List:
        child.remove(child2)
    # filename
    m  = ET.SubElement(child, 'string', {'name': 'filename', 'value': envInfo['envPath']})
    m.tail = '\n        ' if addTail else ''
    m  = ET.SubElement(child, 'float', {'name': 'scale', 'value': envInfo['scale']})
    m.tail = '\n    ' if addTail else ''

def addBsdfNode(root, matInfo):
    # add a bsdf node as the last child of root
    child = ET.SubElement(root, 'bsdf', {'id': matInfo['bsdf_id'], 'type': 'microfacet'})
    addBsdfContentNode(child, matInfo, addTail=False)

def addShapeNode(root, objInfo):
    # add a shape node as the last child of root
    child = ET.SubElement(root, 'shape', {'id': objInfo['shape_id'], 'type': 'obj'})
    addShapeContentNode(child, objInfo, addTail=False)

def addSensorNode(root, sensorInfo):
    # add a sensor node as the last child of root
    child = ET.SubElement(root, 'sensor', {'type': 'perspective'})
    addSensorContentNode(child, sensorInfo, addTail=False)

def addEmitterNode(root, envInfo):
    child = ET.SubElement(root, 'emitter', {'type': 'envmap'})
    addEmitterContentNode(child, envInfo, addTail=False)

def createXmlFromInfoDict(xmlFileOut, xmlInfoDict):
    objInfoDict = xmlInfoDict['shape']
    matInfoDict = xmlInfoDict['bsdf']
    sensorInfo = xmlInfoDict['sensor']
    envInfo = xmlInfoDict['emitter']

    root = ET.Element('scene', {'version': '0.5.0'})
    _    = ET.SubElement(root, 'integrator', {'type': 'path'})
    matDict = {}
    for matID, matInfo in matInfoDict.items():
        matDict[matInfo['bsdf_id']] = matInfo

    addedMatName = []
    for objID, objInfo in objInfoDict.items():
        # add bsdf nodes
        if not (objInfo['isEmitter'] or objInfo['isContainer']):
            for mat in objInfo['matList']:
                if mat not in addedMatName: # add if new material
                    assert(mat in matDict.keys())
                    addBsdfNode(root, matDict[mat])
                    addedMatName.append(mat)
        # add shape node
        addShapeNode(root, objInfo)

    addSensorNode(root, sensorInfo)

    addEmitterNode(root, envInfo)

    if not osp.exists(osp.dirname(xmlFileOut)):
        os.system('mkdir -p %s' % osp.dirname(xmlFileOut))

    indent(root)
    tree = ET.ElementTree(root)
    tree.write(xmlFileOut)

    return xmlFileOut

##### Treat each object separately to enable different materials #####
def saveNewModelIdFileAndXmlFile(xmlFile, meshSaveDir, modelIdFileNew=None, xmlFileNew=None):
    if not osp.exists(meshSaveDir):
        os.system('mkdir -p %s' % meshSaveDir)
    # 1. Generate new modelID file
    # 2. Save new xml file with white materials
    initUVScaleDict = getInitUVScaleDict(xmlFile)
    xmlInfoDict = getXmlInfoDict(xmlFile)
    objInfoDict = xmlInfoDict['shape']
    matInfoDict = xmlInfoDict['bsdf']
    matInfoDict2 = {}
    for matID, matInfo in matInfoDict.items():
        matInfoDict2[matInfo['bsdf_id']] = matInfo
    
    objInfoDictNew = {}
    matInfoDictNew = {}
    shapeCntDict = {}

    for objID in sorted(objInfoDict.keys()):
        objInfo = objInfoDict[objID]
        objInfoNew = copy.deepcopy(objInfo)

        if not (objInfo['isEmitter'] or objInfo['isContainer']):
            # Adjust objName
            objName = objInfo['shape_id']
            if objName in shapeCntDict.keys():
                shapeCntDict[objName] += 1
            else:
                shapeCntDict[objName] = 0
            if 'scene' not in objName:
                tag = '-%02d' % shapeCntDict[objName]
            else:
                tag = ''
            objNameNew = objName.replace('_object', '%s_object' % tag)
            objInfoNew['shape_id'] = objNameNew

            # Update new objPath
            objPath = objInfo['objPath']
            objPathNew = osp.join(meshSaveDir, '%s.obj' % objNameNew.replace('_object', ''))
            objInfoNew['objPath'] = objPathNew

            # check if vt exists
            hasVt = checkVtExist(objPath)
            matList = objInfo['matList']
            matListNew = []
            if hasVt:
                # Modify OBJ usemtl info and Save new OBJ
                # Adjust UV Scale
                matUVDict = {}
                matNameMapping = {}
                for mat in matList:
                    initScale = float(initUVScaleDict[objID][mat])
                    matUVDict[mat] = {'scaleU': initScale, 'scaleV': initScale, 'rot': 0, 'transU': 0, 'transV': 0}
                    partStr = mat.split('_')[-1]
                    matNew = mat.replace('_%s' % partStr, '%s_%s' % (tag, partStr))
                    matNameMapping[matNew] = mat
                    matListNew.append(matNew)
                    # add to new matInfoDict
                    matID = len(matInfoDictNew.keys())
                    matInfoNew = copy.deepcopy(matInfoDict2[mat])
                    matInfoNew['bsdf_id'] = matNew
                    matInfoNew['uvScale'] = '1.000' # set uv scale in xml to 1
                    matInfoDictNew[matID] = matInfoNew
                saveObjWithUvAdjusted(objPath, matUVDict, objPathNew, matNameMapping=matNameMapping)
            else:
                os.system('cp %s %s' % (objPath, objPathNew))
                mat = matList[0]
                matNew = mat.replace('_%s' % partStr, '%s_%s' % (tag, partStr))
                modifyMaterialName(objPathNew, matNew)

            # Update new material names
            objInfoNew['matList'] = matListNew

        objInfoDictNew[objID] = objInfoNew

    # Save new XML
    if xmlFileNew is None:
        xmlFileNew = xmlFile.replace('.xml', '_uv.xml')

    xmlInfoDict['shape'] = objInfoDictNew
    xmlInfoDict['bsdf'] = matInfoDictNew
    xmlFileNew = createXmlFromInfoDict(xmlFileOut=xmlFileNew, xmlInfoDict=xmlInfoDict)

    # Save new models.txt
    if modelIdFileNew is None:
        modelIdFileNew = osp.join(meshSaveDir, 'models.txt')
    numObj = len(objInfoDictNew.keys())
    with open(modelIdFileNew, 'w') as f:
        f.write('%d\n' % numObj)
    objCnt = 0
    for objID, objInfo in objInfoDictNew.items():
        if objInfo['isContainer']:
            continue # skip the objs which don't need the part masks
        objName = objInfo['shape_id']
        strCnt = len(objName.split('_'))
        c2 = objName.split('_')[strCnt-2]
        c1 = '_'.join(objName.split('_')[0:strCnt-2])
        line = '%s %s %d\n' % (c1, c2, objCnt)
        objCnt += 1
        with open(modelIdFileNew, 'a') as f:
            f.write(line)

    return modelIdFileNew, xmlFileNew

def updateXmlNewMatAndShapePath(xmlFileOld, xmlFileNew, matSavePathDict, objInfoDict, isFast=False, nSample=2048):
    # Use after collecting new matSavePathDict and new objInfoDict
    xmlInfoDict = getXmlInfoDict(xmlFileOld)
    matInfoDict = xmlInfoDict['bsdf']
    for matID, matInfo in matInfoDict.items():
        bsdfStrId = matInfo['bsdf_id']
        if bsdfStrId in matSavePathDict.keys():
            matInfoDict[matID]['isHomo'] = False
            for idx, matParam in enumerate(['albedo', 'normal', 'roughness']):
                # newPath = osp.join(matSaveDir, matSavePathDict[bsdfStrId][idx] )
                # matInfoDict[matID][matParam] = newPath
                matInfoDict[matID][matParam] = matSavePathDict[bsdfStrId][idx]
        matInfoDict[matID]['albedoScale'] = '1.000 1.000 1.000'
        matInfoDict[matID]['roughnessScale'] = '1.000'
        matInfoDict[matID]['uvScale'] = '1.000'

    xmlInfoDict['shape'] = objInfoDict

    nSample = '16' if isFast else str(nSample)
    sensorInfoDict = xmlInfoDict['sensor']
    sensorInfoDict['sample_count'] = nSample

    # xmlFileNew = updateXML(xmlFileOld, xmlFileNew=xmlFileNew, matInfoDict=matInfoDict, objInfoDict=objInfoDict, \
    #             envInfoDict=None, nSample=nSample, imWidth=None, imHeight=None)
    xmlFileNew = createXmlFromInfoDict(xmlFileNew, xmlInfoDict)
    return xmlFileNew


def genSingleLightXMLs(xmlDir, layoutMeshDir, scene):
    inXml = osp.join(xmlDir, 'main_median.xml')
    noLightXml = osp.join(xmlDir, 'main_noLight.xml')

    tree = ET.parse(inXml)
    root = tree.getroot()
    # remove ceiling_lamp                
    ceilingLightNodeList = []
    for child in root: 
        if child.tag == 'shape' and 'ceiling_lamp' in child.attrib['id']:
            #for child2 in child:
                # if child2.tag == 'emitter':
                #     emitterNode = copy.deepcopy(child2)
                #     emitterNode[0].set('value', '100.0 100.0 100.0')
            ceilingLightNodeList.append(child)
            #root.remove(child)
    for cl in ceilingLightNodeList:
        root.remove(cl)
    
    for child in root:
        # remove envmap --> adjust to an extremely small value
        if child.tag == 'emitter' and child.attrib['type'] == 'envmap':
            for child2 in child:
                if child2.attrib['name'] == 'scale':
                    child2.set('value', '0.000001' )
        # make the film size to 320 x 240
        if child.tag == 'sensor':
            for child2 in child:
                if child2.tag == 'film':
                    for child3 in child2:
                        if child3.attrib['name'] == 'width':
                            child3.set('value', '320')
                        elif child3.attrib['name'] == 'height':
                            child3.set('value', '240')

    tree.write(noLightXml)
    print('Saved xml file with no light at %s' % noLightXml)

    def getCenterFromMesh(mesh):
        vList = []
        with open(mesh, 'r') as f:
            for line in f.readlines():
                vals = line.strip().split(' ')
                if vals[0] == 'v':
                    vList.append(np.array([float(vals[1]), float(vals[2]), float(vals[3])]))
        center = (np.amax(np.stack(vList, axis=0), axis=0) + np.amin(np.stack(vList, axis=0), axis=0) ) / 2.0
        return center
    
    def applyTransToPoint(pt, opList, attribList):
        for idx, op in enumerate(opList):
            attrib = attribList[idx]
            if op == 'translate':
                disp = np.array([attrib['x'], attrib['y'], attrib['z']])
                pt = pt + disp
            elif op == 'scale':
                scale = np.array([attrib['x'], attrib['y'], attrib['z']])
                pt = pt * scale
            elif op == 'rotate':
                theta = attrib['angle'] / 180 * np.pi
                axis  = np.array([attrib['x'], attrib['y'], attrib['z']])
                axis  = axis / np.linalg.norm(axis)
                rotMat0 = np.eye(3) # cos term
                rotMat1 = np.array([[0.0     , -axis[2], axis[1] ], 
                                    [axis[2] , 0.0     , -axis[0]], 
                                    [-axis[1], axis[0] , 0.0     ]]) # sin term
                rotMat2 = np.matmul(axis[:, np.newaxis], axis[np.newaxis, :])
                rotMat  = np.cos(theta) * rotMat0 + np.sin(theta) * rotMat1 + (1-np.cos(theta)) * rotMat2
                pt      = np.matmul(rotMat, pt[:, np.newaxis])[:, 0]
        return pt

    # add single ceiling light xml
    ceilingLightList = glob.glob(osp.join(layoutMeshDir, '%s_light*.obj' % (scene) ) )
    for cl in sorted(ceilingLightList):
        lName = (cl.split('_')[-1].split('.')[0]).replace('light', 'Light') # Light#
        singleXml = osp.join(xmlDir, 'main_c%s.xml' % lName )
        print('Adding a single ceiling light to %s' % singleXml)
        treeNew = ET.parse(noLightXml)
        rootNew = treeNew.getroot()
        cnt = 0
        for child in rootNew:
            if child.tag == 'shape' and child.attrib['id'] == '%s_object' % scene:
                clNode = copy.deepcopy(child)
                clNode.set('id', '%s_ceiling%s' % (scene, lName))
                refNodeList = []
                for child2 in clNode:
                    #print(child2.tag, ' text: ', child2.text, ' tail: ', child2.tail)
                    if child2.tag == 'string' and child2.attrib['name'] == 'filename':
                        roomMesh = child2.attrib['value']
                        child2.set('value', cl)
                    if child2.tag == 'transform':
                        opList = []
                        attribList = []
                        for child3 in child2:
                            op = child3.tag
                            x = float(child3.attrib['x'])
                            y = float(child3.attrib['y'])
                            z = float(child3.attrib['z'])
                            if op == 'scale' or op == 'translate':
                                attrib = {'x': x, 'y': y, 'z': z}
                            elif op == 'rotate':
                                angle = float(child3.attrib['angle'])
                                attrib = {'angle': angle, 'x': x, 'y': y, 'z': z}
                            else:
                                print('Invalid transform operation!')
                                assert(False)
                            opList.append(op)
                            attribList.append(attrib)
                    if child2.tag == 'ref':
                        #print(child2.attrib['id'])
                        lastTail = child2.tail
                        refNodeList.append(child2)
                        #clNode.remove(child2)
                for rr in refNodeList:
                    clNode.remove(rr)
                clNode[-1].tail = lastTail

                emitterNode = ET.Element('emitter', {'type': 'area'})
                emitterNode.text = '\n            '
                rgbNode     = ET.SubElement(emitterNode, 'rgb', {'value': '300.0 300.0 300.0'})
                rgbNode.tail = '\n        '
                emitterNode.tail = '\n        '
                # m1   = ET.SubElement(root, 'graph', {'name': 'GRAPH_NAME'})
                # emitterNode = copy.deepcopy(child2)
                # emitterNode[0].set('value', '100.0 100.0 100.0')
                clNode.insert(1, emitterNode)
                rootNew.insert(cnt, clNode)
                break
            cnt += 1
        treeNew.write(singleXml)
        print('Write a new xml file for single ceiling light: %s' % singleXml)
    
    # adjust outside envmap scale
    singleXml = osp.join(xmlDir, 'main_eLight.xml')
    treeNew = ET.parse(noLightXml)
    rootNew = treeNew.getroot()
    for child in rootNew:
        if child.tag == 'emitter' and child.attrib['type'] == 'envmap':
            for child2 in child:
                if child2.attrib['name'] == 'scale':
                    child2.set('value', '1.0' )
    treeNew.write(singleXml)
    print('Write a new xml file for env light: %s' % singleXml)


def genSingleLightAreaXMLs(cfg, emitterList):
    # For provided Area lights, e.g. Artist HQ scenes

    # inXml = osp.join(xmlDir, 'main_median.xml')
    inXml = cfg.xmlFile
    noLightXml = cfg.noLightXmlFile

    tree = ET.parse(inXml)
    root = tree.getroot()
    
    for child in root:
        # remove envmap --> adjust to an extremely small value
        if child.tag == 'emitter' and child.attrib['type'] == 'envmap':
            for child2 in child:
                if child2.attrib['name'] == 'scale':
                    child2.set('value', '0.000001' )
        # make the film size to 320 x 240
        if child.tag == 'sensor':
            for child2 in child:
                if child2.tag == 'film':
                    for child3 in child2:
                        if child3.attrib['name'] == 'width':
                            child3.set('value', '%d' % cfg.singleLightImgWidth)
                        elif child3.attrib['name'] == 'height':
                            child3.set('value', '%d' % cfg.singleLightImgHeight)

    tree.write(noLightXml)
    print('Saved xml file with no light at %s' % noLightXml)

    # save single area light xmls
    for idx, emitter in enumerate(emitterList):
        singleXml = cfg.areaLightXmlFile % idx
        print('Adding a single ceiling light to %s' % singleXml)
        treeNew = ET.parse(noLightXml)
        rootNew = treeNew.getroot()
        area = emitter['area']
        intensity = min(100.0, 12.5/area)
        objInfo = {'shape_id': 'areaLight%d_00_object' % idx, 'objPath': emitter['objPath'], 'matList': [], \
                    'isEmitter': True, 'emitterVal': '%f %f %f' % (intensity, intensity, intensity), \
                    'transform_opList': ['scale'], 'transform_attribList': [{'x': '1.000000', 'y': '1.000000', 'z': '1.000000'}], \
                    'isAlignedLight': False, 'isContainer': False}
        addShapeNode(rootNew, objInfo)

        treeNew.write(singleXml)
        print('Write a new xml file for single ceiling light: %s' % singleXml)
    
    # adjust outside envmap scale
    singleXml = cfg.eLightXmlFile
    treeNew = ET.parse(noLightXml)
    rootNew = treeNew.getroot()
    for child in rootNew:
        if child.tag == 'emitter' and child.attrib['type'] == 'envmap':
            for child2 in child:
                if child2.attrib['name'] == 'scale':
                    child2.set('value', '1.0' )
    treeNew.write(singleXml)
    print('Write a new xml file for env light: %s' % singleXml)

