import numpy as np
import struct
import os.path as osp
import cv2
import xml.etree.ElementTree as ET
import os
from PIL import Image
import torch as th
import yaml
# import pymeshlab

def load_cfg(cfg_root):
    # load config file
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    with open(cfg_root, "r") as stream:
        d1 = yaml.safe_load(stream)

    cfg_root2 = d1['rootConfig']
    with open(cfg_root2, "r") as stream:
        d2 = yaml.safe_load(stream)

    # # append repoDir
    # for k, v in d2.items():
    #     d1[k] = osp.join(repoDir, v)
    # d1['repoDir'] = repoDir

    d2.update(d1)
    cfg = Struct(**d2)

    cfg.repoDir = os.getenv('REPO_DIR') if os.getenv('REPO_DIR') is not None else \
                    os.getcwd()
    cfg.logDir   = osp.join(cfg.repoDir, 'logs')
    
    cfg.inputDir = osp.join(cfg.repoDir, cfg.inputDir) if cfg.inputDir[0] != '/' else cfg.inputDir
    cfg.outputDir = osp.join(cfg.repoDir, cfg.outputDir) if cfg.outputDir[0] != '/' else cfg.outputDir

    # USER = os.getenv('API_USER')
    # if [ "$REPODIR" = "" ]; then repoDir=$PWD; else repoDir=$REPODIR; fi
    # print(cfg_root2)
    cfg.inputDir = osp.join(cfg.inputDir, cfg.dataMode)
    cfg.outputDir = osp.join(cfg.outputDir, cfg.dataMode)

    # define and create preprocess folders
    cfg.scene                   = 'scene%s' % cfg.sceneId
    cfg.sceneDir                = osp.join(cfg.outputDir    , cfg.scene)
    cfg.preprocessDir           = osp.join(cfg.sceneDir     , 'preprocess')
    cfg.materialDir             = osp.join(cfg.sceneDir     , 'materials')
    cfg.partIdItem, cfg.normalItem, cfg.uvItem, cfg.depthItem = 'partId', 'normal', 'uvCoord', 'depth'
    cfg.renderDict = {'partId': {'tag': cfg.renderPartId, 'fn': 'imcadmatobj', 'ext': 'dat'}, 
                        'normal': {'tag': cfg.renderNormal, 'fn': 'imnormal', 'ext': 'png'}, 
                        'uvCoord': {'tag': cfg.renderUV, 'fn': 'imuvcoord', 'ext': 'dat'},
                        'depth': {'tag': cfg.renderDepth, 'fn': 'imdepth', 'ext': 'dat'} }
    cfg.partIdRenderDir         = osp.join(cfg.preprocessDir, cfg.partIdItem)
    cfg.normalRenderDir         = osp.join(cfg.preprocessDir, cfg.normalItem)
    cfg.uvRenderDir             = osp.join(cfg.preprocessDir, cfg.uvItem)
    cfg.depthRenderDir          = osp.join(cfg.preprocessDir, cfg.depthItem)
    cfg.partIdPathById          = osp.join(cfg.partIdRenderDir, \
        cfg.renderDict[cfg.partIdItem]['fn'] + '_%d.' + cfg.renderDict[cfg.partIdItem]['ext'])
    cfg.normalPathById          = osp.join(cfg.normalRenderDir, \
        cfg.renderDict[cfg.normalItem]['fn'] + '_%d.' + cfg.renderDict[cfg.normalItem]['ext'])
    cfg.uvPathById              = osp.join(cfg.uvRenderDir, \
        cfg.renderDict[cfg.uvItem]['fn'] + '_%d.' + cfg.renderDict[cfg.uvItem]['ext'])
    cfg.depthPathById           = osp.join(cfg.depthRenderDir, \
        cfg.renderDict[cfg.depthItem]['fn'] + '_%d.' + cfg.renderDict[cfg.depthItem]['ext'])
    cfg.rawRenderDir            = osp.join(cfg.preprocessDir, 'renderRaw')
    cfg.photoDir                = osp.join(cfg.preprocessDir, 'photo')
    cfg.photoOriById            = osp.join(cfg.photoDir     , '%d.png') # idx
    cfg.invRenderDir            = osp.join(cfg.preprocessDir, 'invRender')
    cfg.invRenderEnvmapById     = osp.join(cfg.invRenderDir , '%d_envmap1.npz') # idx
    cfg.invRenderNormalById     = osp.join(cfg.invRenderDir , '%d_normal1.npy') # idx
    cfg.invRenderAlbedoById     = osp.join(cfg.invRenderDir , '%d_albedoBS1.png') # idx
    cfg.invRenderRoughById      = osp.join(cfg.invRenderDir , '%d_roughBS1.png') # idx
    cfg.allCamFile              = osp.join(cfg.preprocessDir, 'cam_all.txt') # n views
    cfg.allCamIdFile            = osp.join(cfg.preprocessDir, 'camId_all.txt') # n views 
    cfg.allCamDir               = osp.join(cfg.preprocessDir, 'cam_all')
    cfg.sampledCamFile          = osp.join(cfg.preprocessDir, 'cam_sampled.txt')
    cfg.sampledCamIdFile        = osp.join(cfg.preprocessDir, 'camId_sampled.txt')
    cfg.selectedCamFile         = osp.join(cfg.preprocessDir, 'cam_selected.txt')
    cfg.selectedCamIdFile       = osp.join(cfg.preprocessDir, 'camId_selected.txt') 
    cfg.selectedViewFile        = osp.join(cfg.preprocessDir, 'selectedViewDict.txt')
    cfg.selectedGraphFile       = osp.join(cfg.preprocessDir, 'selectedGraphDict.txt')
    cfg.selectedSmallDictFile   = osp.join(cfg.preprocessDir, 'selectedViewSmallMaskDict.txt')
    cfg.selectedNoMaskDictFile  = osp.join(cfg.preprocessDir, 'selectedViewNoMaskDict.txt')
    cfg.selectedFullDictFile    = osp.join(cfg.preprocessDir, 'selectedViewFullDict.txt')
    cfg.meshSaveDir             = osp.join(cfg.preprocessDir, 'meshes')
    cfg.modelIdFile             = osp.join(cfg.meshSaveDir  , 'models.txt')
    cfg.seedFile                = osp.join(cfg.preprocessDir, 'seed.txt')
    # cfg.bsdfIdDictFile          = osp.join(cfg.preprocessDir, 'bsdfIdDict.txt')
    cfg.xmlDir                  = osp.join(cfg.preprocessDir, 'xml')
    cfg.xmlFile                 = osp.join(cfg.xmlDir       , 'main.xml')
    cfg.xmlWhiteFile            = osp.join(cfg.xmlDir       , 'main_white.xml')
    cfg.xmlMedianFile           = osp.join(cfg.xmlDir       , 'main_median.xml')
    cfg.xmlInvRenderFile        = osp.join(cfg.xmlDir       , 'main_invrender.xml')
    cfg.noLightXmlFile          = osp.join(cfg.xmlDir       , 'main_noLight.xml')
    cfg.areaLightXmlFile        = osp.join(cfg.xmlDir       , 'main_areaLight%d.xml') # % idx
    cfg.eLightXmlFile           = osp.join(cfg.xmlDir       , 'main_eLight.xml')
    cfg.envFile                 = osp.join(cfg.preprocessDir, 'env.hdr')
    cfg.initPerPixelImListFile  = osp.join(cfg.preprocessDir, 'initPerPixelImList.txt')
    cfg.outputDirByName         = osp.join(cfg.preprocessDir, 'output', '%s') # material name
    cfg.photoByNameId           = osp.join(cfg.outputDirByName, '%d.jpg') # material namd + view idx
    cfg.maskPhotoByNameId       = osp.join(cfg.outputDirByName, '%d_maskSN.png') # material namd + view idx
    cfg.maskPhotoJointByNameId  = osp.join(cfg.outputDirByName, '%d_maskSNJoint.png') # material namd + view idx
    cfg.maskPhotoWeightByNameId = osp.join(cfg.outputDirByName, '%d_maskSNWeight.png') # material namd + view idx
    cfg.maskGeoByNameId         = osp.join(cfg.outputDirByName, '%d_maskOR.png') # material namd + view idx
    cfg.maskGeoFromPhotoByNameId        = osp.join(cfg.outputDirByName, '%d_maskORfromSN.png') # material namd + view idx
    cfg.maskGeoFromPhotoJointByNameId   = osp.join(cfg.outputDirByName, '%d_maskORfromSNJoint.png') # material namd + view idx
    cfg.maskGeoFromPhotoWeightByNameId  = osp.join(cfg.outputDirByName, '%d_maskORfromSNWeight.png') # material namd + view idx
    cfg.albedoGeoFromPhotoByNameId      = osp.join(cfg.outputDirByName, '%d_albedoORfromSN.png') # material namd + view idx
    cfg.roughGeoFromPhotoByNameId       = osp.join(cfg.outputDirByName, '%d_roughORfromSN.png') # material namd + view idx
    cfg.targetGeoFromPhotoByNameId      = osp.join(cfg.outputDirByName, '%d_targetORfromSN.png') # material namd + view idx
    cfg.targetBlendGeoFromPhotoByNameId = osp.join(cfg.outputDirByName, '%d_blendTargetORfromSN.png') # material namd + view idx
    cfg.uvWarpedByNameId        = osp.join(cfg.outputDirByName, '%d_imuvcoordWarped.npy') # material namd + view idx
    # cfg.uvWarpedVisByNameId     = osp.join(cfg.outputDirByName, '%d_imuvcoordWarped.png') # material namd + view idx
    # cfg.uvUnwarpedVisByNameId   = osp.join(cfg.outputDirByName, '%d_imuvcoordUnwarped.png') # material namd + view idx
    cfg.warpInfoByNameId        = osp.join(cfg.outputDirByName, '%d_warpInfo.txt') # material namd + view idx

    cfg.matWhiteDirByName       = osp.join(cfg.materialDir  , '%s', 'white') # material name
    cfg.matMedianDirByName      = osp.join(cfg.materialDir  , '%s', 'median') # material name
    cfg.matInvRenderDirByName   = osp.join(cfg.materialDir  , '%s', 'invrender') # material name
    cfg.matPhotoSceneDirByName  = osp.join(cfg.materialDir  , '%s', 'photoscene') # material name
    cfg.matFirstDirByName       = osp.join(cfg.materialDir  , '%s', 'optMat') # material name
    cfg.matPathByNameTypeGraphParamMap = osp.join(cfg.materialDir, '%s', '%s', '%s-%s', '%s.png') # mat name, type, graph-param, map.png
    cfg.program                 = osp.join(cfg.rendererRoot , 'OptixRenderer/src/bin/optixRenderer')
    cfg.programMatPart          = osp.join(cfg.rendererRoot , 'OptixRenderer_MatPart/src/bin/optixRenderer')
    cfg.programUV               = osp.join(cfg.rendererRoot , 'OptixRenderer_UVcoord/src/bin/optixRenderer')
    cfg.programLight            = osp.join(cfg.rendererRoot , 'OptixRenderer_Light/src/bin/optixRenderer')
    cfg.graphListFile           = osp.join('txt', 'graphList.txt')
    if cfg.dataMode == 'total3d':
        cfg.total3dInputDir  = osp.join(cfg.inputDir, 'inputs')
        cfg.total3dInputFile = osp.join(cfg.inputDir, 'inputs', '%s.pkl' % cfg.sceneId)
        cfg.total3dOutputDir = osp.join(cfg.inputDir, 'outputs', str(cfg.sceneId)) # total 3d preprocessed result
        cfg.srcCamFile       = osp.join(cfg.preprocessDir, 'cam_src.txt')
        cfg.panopticSrcDir   = osp.join(cfg.preprocessDir, 'panoptic')
        cfg.meshSrcSaveDir   = osp.join(cfg.preprocessDir, 'meshesSrc') # Input
        cfg.photoSrcDir      = osp.join(cfg.preprocessDir, 'photoSrc')
        cfg.photoSavePath    = osp.join(cfg.photoSrcDir, '0.png') # Invrender net assumes .png
        # cfg.semLabDir        = osp.join(cfg.panopticSrcDir, 'semantic') # vId.png
        cfg.semLabById       = osp.join(cfg.panopticSrcDir, 'semantic', '%d.png') # idx
        # cfg.insLabDir        = osp.join(cfg.panopticSrcDir, 'instance') # vId.png
        cfg.insLabById          = osp.join(cfg.panopticSrcDir, 'instance', '%d.png') # idx
        cfg.labelMapFile     = osp.join('txt', 'sunrgbd2ade20kAdd1.txt')
        cfg.gtMaskDir        = None
        cfg.xmlSrcFile       = osp.join(cfg.xmlDir     , 'main_src.xml')

    if cfg.gtMaskDir is not None: # use provided part masks
        assert(osp.exists(cfg.gtMaskDir))
        cfg.gtMaskByNameId = osp.join(cfg.gtMaskDir, '%s', '%d.png') # material namd + view idx
    

    # Command prefix
    cfg.total3dPreprocessCmdPre = 'python3 utils/process_total3d.py --input_pkl %s' # cfg.total3dInputFile
    cfg.total3dRunCmdPre \
        = 'cd %s;' % cfg.total3dRoot \
        + 'python3 main.py configs/total3d.yaml --mode demo --demo_path %s ' \
        + '> %s; ' % osp.join(cfg.logDir, 'total3D.txt') \
        + 'cd %s' % cfg.repoDir # total 3d dir, data dir, repo
    cfg.blenderApplyUvCmdPre    = 'blender -b -P photoscene/applyUV.py -- %s %s > tmp.txt; rm tmp.txt' # -- inputPath, outputPath
    cfg.maskFormerCmdPre \
        = 'cd %s; python3 demo/demo_photoscene.py ' %  cfg.maskformerRoot \
        + '--config-file configs/ade20k-150-panoptic/maskformer_panoptic_R101_bs16_720k.yaml ' \
        + '--input %s --output %s --opts MODEL.WEIGHTS demo/model_final_0b3cc8.pkl ' \
        + '> %s; ' % osp.join(cfg.logDir, 'maskFormer.txt') \
        + 'cd %s' % cfg.repoDir
        # maskformer dir, ..., repo
    cfg.renderPartIdCmdPre      = '%s -f %s -c %s -o %s -m 7 --modelIdFile %s' + ' > %s' % osp.join(cfg.logDir, 'renderPartId.txt')
    cfg.renderNormalCmdPre      = '%s -f %s -c %s -o %s -m 2' + ' > %s' % osp.join(cfg.logDir, 'renderNormal.txt')
    cfg.renderUvCmdPre          = '%s -f %s -c %s -o %s -m 7' + ' > %s' % osp.join(cfg.logDir, 'renderUV.txt')
    cfg.renderDepthCmdPre       = '%s -f %s -c %s -o %s -m 5' + ' > %s' % osp.join(cfg.logDir, 'renderDepth.txt')
    cfg.invrenderCmdPre         \
        = 'cd %s;' % cfg.invrenderRoot \
        + ' python3 -W ignore testReal.py --cuda --dataRoot %s' \
        + ' --imList %s' \
        + ' --testRoot %s --isLight --isBS --level 2' \
        + ' --experiment0 check_cascade0_w320_h240 --nepoch0 14' \
        + ' --experimentLight0 check_cascadeLight0_sg12_offset1.0 --nepochLight0 10' \
        + ' --experimentBS0 checkBs_cascade0_w320_h240' \
        + ' --experiment1 check_cascade1_w320_h240 --nepoch1 7' \
        + ' --experimentLight1 check_cascadeLight1_sg12_offset1.0 --nepochLight1 10' \
        + ' --experimentBS1 checkBs_cascade1_w320_h240 ' + '> %s; ' % osp.join(cfg.logDir, 'invRender.txt') \
        + 'cd %s' % cfg.repoDir

    # Commands
    if not osp.exists(cfg.logDir):
        os.makedirs(cfg.logDir)
    cfg.renderPartIdCmd = cfg.renderPartIdCmdPre % \
        (cfg.programMatPart, cfg.xmlFile, cfg.allCamFile, osp.join(cfg.partIdRenderDir, 'im.hdr'), cfg.modelIdFile )
    cfg.renderNormalCmd = cfg.renderNormalCmdPre % \
        (cfg.program, cfg.xmlFile, cfg.allCamFile, osp.join(cfg.normalRenderDir, 'im.hdr') )
    cfg.renderUvCmd     = cfg.renderUvCmdPre % \
        (cfg.programUV, cfg.xmlFile, cfg.allCamFile, osp.join(cfg.uvRenderDir, 'im.hdr') )
    cfg.renderDepthCmd  = cfg.renderDepthCmdPre % \
        (cfg.program, cfg.xmlFile, cfg.allCamFile, osp.join(cfg.depthRenderDir, 'im.hdr') )
    cfg.invrenderCmd    = cfg.invrenderCmdPre % \
        (cfg.photoSrcDir, cfg.initPerPixelImListFile, cfg.invRenderDir)
    if cfg.dataMode == 'total3d':
        cfg.total3dPreprocessCmd = cfg.total3dPreprocessCmdPre % cfg.total3dInputFile
        demo_path = osp.join(cfg.total3dInputDir, str(cfg.sceneId))
        cfg.total3dRunCmd    = cfg.total3dRunCmdPre % (demo_path)
        cfg.genPanopticCmd   = cfg.maskFormerCmdPre % (cfg.photoSavePath, cfg.panopticSrcDir)

    return cfg

def loadBinary(imName, channels=1, dtype=np.float32, if_resize=False, imWidth=None, imHeight=None):
    assert dtype in [
        np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
    if not(osp.isfile(imName)):
        print(imName)
        assert(False)
    with open(imName, 'rb') as fIn:
        hBuffer = fIn.read(4)
        height = struct.unpack('i', hBuffer)[0]
        wBuffer = fIn.read(4)
        width = struct.unpack('i', wBuffer)[0]
        dBuffer = fIn.read(4 * channels * width * height)
        if dtype == np.float32:
            decode_char = 'f'
        elif dtype == np.int32:
            decode_char = 'i'
        depth = np.asarray(struct.unpack(
            decode_char * channels * height * width, dBuffer), dtype=dtype) # 0.02 ms ~ 0.1 ms, slow
        depth = depth.reshape([height, width, channels])

        if if_resize:
            #t0 = timeit.default_timer()
            # print(self.imWidth, self.imHeight, width, height)
            if dtype == np.float32:
                depth = cv2.resize(
                    depth, (imWidth, imHeight), interpolation=cv2.INTER_AREA)
                #print('Resize float binary: %.4f' % (timeit.default_timer() - t0) )
            elif dtype == np.int32:
                depth = cv2.resize(depth.astype(
                    np.float32), (imWidth, imHeight), interpolation=cv2.INTER_NEAREST)
                depth = depth.astype(np.int32)
                #print('Resize int32 binary: %.4f' % (timeit.default_timer() - t0) )

        depth = np.squeeze(depth)

    return depth[np.newaxis, :, :]

def readModelStrIdDict(modelListPath):
    modelListFile = open(modelListPath, 'r')
    modelStrIdDict = {}
    for i, line in enumerate(modelListFile.readlines()):
        if i == 0:
            continue
        item = line.strip().split(' ')
        modelStr = '%s_%s' % (item[0], item[1])
        modelId  = int(item[2])
        modelStrIdDict[modelStr] = modelId
    return modelStrIdDict

# def readMaterialDict(graphDictFile):
#     matDict = {}
#     with open(graphDictFile, 'r') as f:
#         for line in f.readlines():
#             mat, graph = line.strip().split(' ')
#             if mat is not None:
#                 matDict[mat] = graph
#     return matDict

def readGraphDict(graphDictFile):
    matDict = {}
    with open(graphDictFile, 'r') as f:
        for line in f.readlines():
            mat, graph = line.strip().split(' ')
            if mat is not None:
                matDict[mat] = graph
    return matDict

def readViewDict(selectedViewFile):
    matViewDict = {}
    with open(selectedViewFile, 'r') as f:
        for line in f.readlines():
            m, vId = line.strip().split(' ')
            matViewDict[m] = int(vId)
    # print('Read selected views from %s' % selectedViewFile)

    return matViewDict

def readViewGraphDict(viewDictFile, graphDictFile):
    matDict = {}
    # viewDictFile = osp.join(viewDictDir, 'selectedViewDict.txt') 
    with open(viewDictFile, 'r') as f:
        for line in f.readlines():
            mat, vId = line.strip().split(' ')
            if mat is not None and vId != '-1':
                matDict[mat] = {'vId': int(vId)}
    # graphDictFile = osp.join(viewDictDir, 'selectedGraphDict.txt') 
    with open(graphDictFile, 'r') as f:
        for line in f.readlines():
            mat, graph = line.strip().split(' ')
            if mat is not None and mat in matDict.keys():
                matDict[mat]['graph'] = graph
    return matDict

def genSavePathDict(cfg, vIdDict, baseline, tag, isHomo=False):
    # matSavePathDict = {'scene0001_00_wall': 
    #                         ['leather_small/%s/basecolor/basecolor.png' % tag, \
    #                         'leather_small/%s/normal/normal.png' % tag, \
    #                         'leather_small/%s/roughness/roughness.png' % tag], 
    #                         'scene0001_00_floor': 
    #                         ['leather_small/%s/basecolor/basecolor.png' % tag, \
    #                         'leather_small/%s/normal/normal.png' % tag, \
    #                         'leather_small/%s/roughness/roughness.png' % tag], 
    #                         }
    matSavePathDict = {}
    for partName in vIdDict.keys():
        fileList = []
        isExist = True
        for matParam in ['basecolor', 'normal', 'roughness']:
            if not isHomo:
                graph_name = vIdDict[partName]['graph']
            else:
                graph_name = 'liang_test'
            # fileName = os.path.join(graph_name, tag, matParam, '%s.png' % matParam)
            # fullPath = os.path.join(graphSaveDir, partName, fileName) 
            fullPath = cfg.matPathByNameTypeGraphParamMap % \
                (partName, baseline, graph_name, tag, matParam)
            if not os.path.exists(fullPath):
                print('Warning! %s not exist!' % fullPath)
                isExist = False
            # fileList.append(fileName)
            fileList.append(fullPath)
        if isExist:
            matSavePathDict[partName] = fileList
    return matSavePathDict

def getObjMatVertFaceDict(objPath):
    v  = []
    vn = []
    vt = []
    matFacesListDict = {} # {matName: list of faces}
    matList = []
    with open(objPath, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            vals = line.strip().split()
            if len(vals) == 0:
                continue
            if vals[0] == 'v':
                v.append(np.array([float(vals[1]), float(vals[2]), float(vals[3])]))
            elif vals[0] == 'vn':
                vn.append(np.array([float(vals[1]), float(vals[2]), float(vals[3])]))
            elif vals[0] == 'vt':
                vt.append(np.array([float(vals[1]), float(vals[2])]))
            elif vals[0] == 'usemtl':
                mtlCurr = vals[1]
                matFacesListDict[mtlCurr] = []
                matList.append(mtlCurr)
            elif vals[0] == 'f':
                nItems = len(vals[1].split('/'))
                if nItems == 2:
                    v1, vt1 = vals[1].split('/')
                    v2, vt2 = vals[2].split('/')
                    v3, vt3 = vals[3].split('/')
                    faceDict = {'v1': int(v1), 'vt1': int(vt1), 
                                'v2': int(v2), 'vt2': int(vt2), 
                                'v3': int(v3), 'vt3': int(vt3) }
                elif nItems == 3:
                    v1, vt1, vn1 = vals[1].split('/')
                    v2, vt2, vn2 = vals[2].split('/')
                    v3, vt3, vn3 = vals[3].split('/')
                
                    if len(vt1) > 0 and len(vt2) > 0 and len(vt3) > 0:
                        faceDict = {'v1': int(v1), 'vt1': int(vt1), 'vn1': int(vn1), 
                                    'v2': int(v2), 'vt2': int(vt2), 'vn2': int(vn2), 
                                    'v3': int(v3), 'vt3': int(vt3), 'vn3': int(vn3) }
                    else:
                        faceDict = {'v1': int(v1), 'vn1': int(vn1), 
                                    'v2': int(v2), 'vn2': int(vn2), 
                                    'v3': int(v3), 'vn3': int(vn3) }
                else:
                    print(objPath)
                    print(vals)
                    assert(False)

                matFacesListDict[mtlCurr].append(faceDict)
    return v, vn, vt, matFacesListDict, matList

def saveObjFromMatVertFaceDict(v, vn, vt, matFacesListDict, matList, saveRoot, isOverwrite=False):
    print('Saving new OBJ at %s ... ' % saveRoot, end='')
    if osp.exists(saveRoot):
        print('Exist! ', end='')
        if isOverwrite:
            print('Overwrite!')
        else:
            print('Skip!')
            return saveRoot
    print('')
    ### Save new mesh ###
    if not osp.exists(osp.dirname(saveRoot)):
        os.system('mkdir -p %s' % osp.dirname(saveRoot))
    with open(saveRoot, 'w') as f:
        f.write('')
    # write v
    for idx, vv in enumerate(v):
        line = 'v %f %f %f' % (vv[0], vv[1], vv[2])
        with open(saveRoot, 'a') as f:
            f.write('{}\n'.format(line))
    # write vt
    for idx, vv in enumerate(vt):
        line = 'vt %f %f' % (vv[0], vv[1])
        with open(saveRoot, 'a') as f:
            f.write('{}\n'.format(line))
    # write vn
    for idx, vv in enumerate(vn):
        line = 'vn %f %f %f' % (vv[0], vv[1], vv[2])
        with open(saveRoot, 'a') as f:
            f.write('{}\n'.format(line))
    # write f
    for mat in matList:
        line = 'usemtl %s' % (mat)
        with open(saveRoot, 'a') as f:
            f.write('{}\n'.format(line))
        for face in matFacesListDict[mat]:
            f1, f2, f3 = str(face['v1']), str(face['v2']), str(face['v3'])
            if 'vt1' in face.keys() and 'vt2' in face.keys() and 'vt3' in face.keys():
                vt1, vt2, vt3 = str(face['vt1']), str(face['vt2']), str(face['vt3'])
            else:
                vt1, vt2, vt3 = '', '', ''
            f1, f2, f3 = f1 + '/' + vt1, f2 + '/' + vt2, f3 + '/' + vt3
            if 'vn1' in face.keys() and 'vn2' in face.keys() and 'vn3' in face.keys():
                vn1, vn2, vn3 = str(face['vn1']), str(face['vn2']), str(face['vn3'])
                f1, f2, f3 = f1 + '/' + vn1, f2 + '/' + vn2, f3 + '/' + vn3
            # line = 'f %d/%d/%d %d/%d/%d %d/%d/%d' % (v1, vt1, vn1, v2, vt2, vn2, v3, vt3, vn3)
            line = 'f %s %s %s' % (f1, f2, f3)
            with open(saveRoot, 'a') as f:
                f.write('{}\n'.format(line))
    return saveRoot

def resizeImg(imgPath, targetWidth, outputPath):
    img = cv2.imread(imgPath, -1)
    scale = targetWidth / img.shape[1] # percent of original size
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(outputPath, resized)

def checkImgListExist(imgList):
    for imgPath in imgList:
        if not osp.exists(imgPath):
            print('%s not exist!' % imgPath)
            return False
    return True

# def readNormal(normalFile, refW, refH):
#     normalOriImg  = Image.open(normalFile).convert('RGB').resize((refW, refH), Image.ANTIALIAS)
#     normalOriImg  = np.asarray(normalOriImg) / 255.0
#     normalOri  = normalOriImg * 2 - 1
#     normalOri  = normalOri / np.sqrt(np.sum(normalOri ** 2, axis=2) )[:,:,np.newaxis]
#     normalOri = th.from_numpy(normalOri.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
#     return normalOri

# def loadInvRenderData(cfg, vId, refH, refW):
#     # print('Load input RGB!')
#     initFolder = cfg.photoDir
#     imFile = os.path.join(initFolder, '%d.png' % vId)
#     imOri      = Image.open(imFile).convert('RGB')
#     imOri_resize = np.asarray(imOri.resize((refW, refH), Image.ANTIALIAS), dtype=np.float32) / 255.0
#     imOri      = (np.asarray(imOri, dtype=np.float32) / 255.0 )
#     # imOri      = read_image(imFile)
#     imH, imW, imC = imOri.shape
#     imOri = th.from_numpy(imOri).permute(2, 0, 1).unsqueeze(0)
#     target_ref = th.from_numpy(imOri_resize).permute(2, 0, 1).unsqueeze(0) # 1 x 3 x H x W

#     initFolder = cfg.invRenderDir
#     # print('Load env map!')
#     # envFile = os.path.join(initFolder, '%d_envmap1.npy' % vId)
#     # envmap     = np.load(envFile) # [bn=1, 3, envRow, envCol, self.envHeight, self.envWidth]
#     # envmap     = th.from_numpy(envmap).unsqueeze(0).to(device)# [1, 3, 119, 160, 8, 16]

#     # print('Load init. normal!')
#     normalFile = os.path.join(initFolder, '%d_normal1.png' % vId)
#     normalOri = readNormal(normalFile, refW, refH)
#     #print('Load init. albode!')
#     albedoFile = os.path.join(initFolder, '%d_albedoBS1.png' % vId)
#     albedoOri  = Image.open(albedoFile).convert('RGB').resize((refW, refH), Image.ANTIALIAS)
#     albedoOri  = (np.asarray(albedoOri) / 255.0) ** 2.2
#     imH, imW, _ = albedoOri.shape
#     albedoOri = th.from_numpy(albedoOri).permute(2, 0, 1).unsqueeze(0)

#     #print('Load init. roughness!')
#     roughFile  = os.path.join(initFolder, '%d_roughBS1.png' % vId)
#     roughOri  = Image.open(roughFile).resize((refW, refH), Image.ANTIALIAS)
#     roughOri  = (np.asarray(roughOri) / 255.0)
#     imH, imW = roughOri.shape
#     roughOri = th.from_numpy(roughOri[np.newaxis,np.newaxis,:,:])

#     # print('Load init. depth!')
#     # depthFile = os.path.join(initFolder, '%d_depthBS1.npy' % vId)
#     # depthOri = np.load(depthFile) # 239 x 320
#     # depthOri = th.from_numpy(depthOri[np.newaxis,np.newaxis,:,:]).to(device)
#     # depthOri = th.nn.functional.interpolate(depthOri, size=(refH, refW))

#     invRenderData = {'imOri': imOri,
#                     'im'    : target_ref,
#                     'normal': normalOri,
#                     'albedo': albedoOri,
#                     'rough': roughOri }
#     #print(th.max(albedoOri), th.max(roughOri))
#     return invRenderData

def readNormal(normalFile, refW, refH):
    ext = normalFile.split('.')[-1]
    if ext == 'npy':
        normalOri = cv2.resize(np.load(normalFile), (refW, refH), interpolation=cv2.INTER_AREA)
    else:
        normalOriImg  = Image.open(normalFile).convert('RGB').resize((refW, refH), Image.ANTIALIAS)
        normalOriImg  = np.asarray(normalOriImg) / 255.0
        normalOri  = normalOriImg * 2 - 1
    normalOri  = normalOri / np.sqrt(np.sum(normalOri ** 2, axis=2) )[:,:,np.newaxis]
    normalOri = th.from_numpy(normalOri.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    return normalOri

def loadInvRenderData(cfg, vId, refH, refW, isLoadEnv=False, device='cuda'):
    # initFolder = os.path.join(inDir, 'snColor')
    # imFile = os.path.join(initFolder, '%d.jpg' % vId)
    imFile = cfg.photoOriById % vId
    imOri      = Image.open(imFile).convert('RGB')
    imOri_resize = np.asarray(imOri.resize((refW, refH), Image.ANTIALIAS), dtype=np.float32) / 255.0
    imOri      = (np.asarray(imOri, dtype=np.float32) / 255.0 )
    # imOri      = read_image(imFile)
    imH, imW, imC = imOri.shape
    imOri = th.from_numpy(imOri).permute(2, 0, 1).unsqueeze(0).to(device)
    target_ref = th.from_numpy(imOri_resize).permute(2, 0, 1).unsqueeze(0).to(device) # 1 x 3 x H x W

    # print('Load init. normal!')
    # normalFile = os.path.join(initFolder, '%d_normal1.png' % vId)
    normalFile = cfg.invRenderNormalById % vId
    normalOri = readNormal(normalFile, refW, refH).to(device)

    # albedoFile = os.path.join(initFolder, '%d_albedoBS1.png' % vId)
    albedoFile = cfg.invRenderAlbedoById % vId
    albedoOri  = Image.open(albedoFile).convert('RGB').resize((refW, refH), Image.ANTIALIAS)
    albedoOri  = (np.asarray(albedoOri) / 255.0) ** 2.2
    imH, imW, _ = albedoOri.shape
    albedoOri = th.from_numpy(albedoOri).permute(2, 0, 1).unsqueeze(0)

    #print('Load init. roughness!')
    # roughFile  = os.path.join(initFolder, '%d_roughBS1.png' % vId)
    roughFile = cfg.invRenderRoughById % vId
    roughOri  = Image.open(roughFile).resize((refW, refH), Image.ANTIALIAS)
    roughOri  = (np.asarray(roughOri) / 255.0)
    imH, imW = roughOri.shape
    roughOri = th.from_numpy(roughOri[np.newaxis,np.newaxis,:,:])

    invRenderData = {'imOri': imOri,
                    'im'    : target_ref,
                    'normal': normalOri,
                    'albedo': albedoOri,
                    'rough': roughOri }

    if isLoadEnv:
        # initFolder = os.path.join(inDir, 'invRender')
        # initFolder = cfg.invRenderDir
        # print('Load env map!')
        # envFile = os.path.join(initFolder, '%d_envmap1.npy' % vId)
        envFile = cfg.invRenderEnvmapById % vId
        envmap     = np.transpose(np.load(envFile)['env'], (4, 0, 1, 2, 3)) # [bn=1, 3, envRow, envCol, self.envHeight, self.envWidth]
        envmap     = th.from_numpy(np.asarray(envmap)).unsqueeze(0).to(device)# [1, 3, 119, 160, 8, 16]
        invRenderData['envmap'] = envmap

    return invRenderData

# def convertMesh(meshRoot, outRoot):
#     # PyMeshLab version
#     # segmentation fault for glb files -> no faces exported
#     # ext = meshRoot.split('.')[-1]
#     ms = pymeshlab.MeshSet()
#     ms.load_new_mesh(meshRoot)
#     ms.save_current_mesh(outRoot, save_vertex_coord=True) 

def checkVtExist(objPath):
    if not osp.exists(objPath):
        assert(False)
    with open(objPath, 'r') as f:
        for line in f.readlines():
            if len(line.strip().split(' ')) == 0:
                continue
            first = line.strip().split(' ')[0]
            if first == 'vt':
                return True
    print('UV coordinate for %s not exist!'  % objPath)
    return False

def modifyMaterialName(objPath, bsdfName):
    objPathTemp = objPath.replace('.obj', '_temp.obj')
    hasUsemtl = False
    oldMatName = 'n/a'
    with open(objPathTemp, 'w') as fw:
        with open(objPath, 'r') as f:
            for line in f.readlines():
                if line.strip().split(' ')[0] == 'usemtl':
                    oldMatName = line.strip().split(' ')[1]
                    newline = '\n'
                elif line.strip().split(' ')[0] == 'mtllib':
                    newline = '\n'
                else:
                    if line.strip().split(' ')[0] == 'f' and not hasUsemtl:
                        fw.write('usemtl %s\n' % bsdfName)
                        hasUsemtl = True
                        # print('Change old material name %s to %s' % (oldMatName, bsdfName))
                    newline = line
                fw.write(newline)

    os.system('rm %s' % objPath)
    os.system('mv %s %s' % (objPathTemp, objPath))