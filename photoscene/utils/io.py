import numpy as np
import struct
import os.path as osp
import cv2
import os
from PIL import Image
import torch as th
import yaml


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

    cfg_root3 = d1['baseConfig']
    with open(cfg_root3, "r") as stream:
        d3 = yaml.safe_load(stream)

    d1.update(d2)
    d1.update(d3)
    cfg = Struct(**d1)

    assert (len(cfg.optObjectives) == len(cfg.optObjectivesCoeff) and len(cfg.optObjectives) > 0)
    paramStr = ''
    cfg.optObjectiveDict = {}
    for idx, obj in enumerate(cfg.optObjectives):
        objCoef = cfg.optObjectivesCoeff[idx]
        cfg.optObjectiveDict[obj] = objCoef
        paramStr += '%s-%0.2f' % (obj, objCoef)
        if idx < len(cfg.optObjectives) - 1:
            paramStr += '-'
    cfg.paramStr = paramStr

    cfg.uvConfigDict = {'uvScaleBase': cfg.uvScaleBase,
                        'uvInitScale': cfg.uvInitScale,
                        'uvInitRot': cfg.uvInitRot,
                        'uvInitTrans': cfg.uvInitTrans,
                        'uvScaleExpMin': cfg.uvScaleExpMin,
                        'uvScaleExpMax': cfg.uvScaleExpMax,
                        'uvScaleStepNum': cfg.uvScaleStepNum,
                        'uvRotStepNum': cfg.uvRotStepNum,
                        'uvTransStepNum': cfg.uvTransStepNum}

    cfg.repoDir = os.getenv('REPO_DIR') if os.getenv('REPO_DIR') is not None else \
                    os.getcwd()
    cfg.logDir   = osp.join(cfg.repoDir, 'logs')

    cfg.inputDir = osp.join(cfg.repoDir, cfg.inputDir) if cfg.inputDir[0] != '/' else cfg.inputDir
    cfg.outputDir = osp.join(cfg.repoDir, cfg.outputDir) if cfg.outputDir[0] != '/' else cfg.outputDir

    cfg.inputDir = osp.join(cfg.inputDir, cfg.dataMode)
    cfg.outputDir = osp.join(cfg.outputDir, cfg.dataMode)

    # define and create preprocess folders
    cfg.scene                   = 'scene%s' % cfg.sceneId
    cfg.sceneDir                = osp.join(cfg.outputDir    , cfg.scene)
    cfg.preprocessDir           = osp.join(cfg.sceneDir     , 'preprocess')
    cfg.materialDir             = osp.join(cfg.sceneDir     , 'materials')
    cfg.graphDir                = osp.join(cfg.preprocessDir, 'graphs')
    cfg.partIdItem, cfg.normalItem, cfg.uvItem, cfg.depthItem, cfg.envmapItem, cfg.imageItem = \
        'partId', 'normal', 'uvCoord', 'depth', 'light', 'image'
    cfg.renderDict = {  'partId'    : {'fn': 'imcadmatobj'  , 'ext': 'dat'},
                        'normal'    : {'fn': 'imnormal'     , 'ext': 'png'},
                        'uvCoord'   : {'fn': 'imuvcoord'    , 'ext': 'dat'},
                        'depth'     : {'fn': 'imdepth'      , 'ext': 'dat'},
                        'light'     : {'fn': 'imenv'        , 'ext': 'hdr'},
                        'image'     : {'fn': 'im'           , 'ext': 'rgbe'},
                        'albedo'    : {'fn': 'imbaseColor'  , 'ext': 'png'},
                        'rough'     : {'fn': 'imroughness'  , 'ext': 'png'}}
    cfg.initRenderDir           = osp.join(cfg.preprocessDir, 'renderInit')
    cfg.partIdRenderDir         = osp.join(cfg.initRenderDir, cfg.partIdItem)
    cfg.normalRenderDir         = osp.join(cfg.initRenderDir, cfg.normalItem)
    cfg.uvRenderDir             = osp.join(cfg.initRenderDir, cfg.uvItem)
    cfg.depthRenderDir          = osp.join(cfg.initRenderDir, cfg.depthItem)
    cfg.partIdPathById          = osp.join(cfg.partIdRenderDir,
        cfg.renderDict[cfg.partIdItem]['fn'] + '_%d.' + cfg.renderDict[cfg.partIdItem]['ext'])
    cfg.normalPathById          = osp.join(cfg.normalRenderDir,
        cfg.renderDict[cfg.normalItem]['fn'] + '_%d.' + cfg.renderDict[cfg.normalItem]['ext'])
    cfg.uvPathById              = osp.join(cfg.uvRenderDir,
        cfg.renderDict[cfg.uvItem]['fn'] + '_%d.' + cfg.renderDict[cfg.uvItem]['ext'])
    cfg.depthPathById           = osp.join(cfg.depthRenderDir,
        cfg.renderDict[cfg.depthItem]['fn'] + '_%d.' + cfg.renderDict[cfg.depthItem]['ext'])
    cfg.photoDir                = osp.join(cfg.sceneDir     , 'photo')
    cfg.photoOriById            = osp.join(cfg.photoDir     , '%d.png')  # idx
    cfg.invRenderDir            = osp.join(cfg.preprocessDir, 'invRender')
    cfg.invRenderEnvmapById     = osp.join(cfg.invRenderDir , '%d_envmap1.npz')  # idx
    cfg.invRenderNormalById     = osp.join(cfg.invRenderDir , '%d_normal1.npy')  # idx
    cfg.invRenderAlbedoById     = osp.join(cfg.invRenderDir , '%d_albedoBS1.png')  # idx
    cfg.invRenderRoughById      = osp.join(cfg.invRenderDir , '%d_roughBS1.png')  # idx
    cfg.camDir                  = osp.join(cfg.preprocessDir, 'cameras')
    cfg.allCamFile              = osp.join(cfg.camDir       , 'cam_all.txt')  # n views
    cfg.allCamIdFile            = osp.join(cfg.camDir       , 'camId_all.txt')  # n views
    cfg.allCamDir               = osp.join(cfg.camDir       , 'camAll')
    cfg.sampledCamFile          = osp.join(cfg.camDir       , 'cam_sampled.txt')
    cfg.sampledCamIdFile        = osp.join(cfg.camDir       , 'camId_sampled.txt')
    cfg.selectedCamFile         = osp.join(cfg.camDir       , 'cam_selected.txt')
    cfg.selectedCamIdFile       = osp.join(cfg.camDir       , 'camId_selected.txt')
    cfg.selectedFile            = osp.join(cfg.preprocessDir, 'selected')
    cfg.selectedViewFile        = osp.join(cfg.selectedFile , 'selectedViewDict.txt')
    cfg.selectedGraphFile       = osp.join(cfg.selectedFile , 'selectedGraphDict.txt')
    cfg.selectedSmallDictFile   = osp.join(cfg.selectedFile , 'selectedViewSmallMaskDict.txt')
    cfg.selectedNoMaskDictFile  = osp.join(cfg.selectedFile , 'selectedViewNoMaskDict.txt')
    cfg.selectedFullDictFile    = osp.join(cfg.selectedFile , 'selectedViewFullDict.txt')
    cfg.meshSaveDir             = osp.join(cfg.preprocessDir, 'meshes')
    cfg.modelIdFile             = osp.join(cfg.meshSaveDir  , 'models.txt')
    cfg.meshSaveInitDir         = osp.join(cfg.meshSaveDir  , 'first')
    cfg.meshSaveFinalDir        = osp.join(cfg.sceneDir     , 'meshes')
    cfg.seedFile                = osp.join(cfg.preprocessDir, 'seed.txt')
    cfg.xmlDir                  = osp.join(cfg.preprocessDir, 'xmlInit')
    cfg.xmlFile                 = osp.join(cfg.xmlDir       , 'main.xml')
    cfg.xmlWhiteFile            = osp.join(cfg.xmlDir       , 'main_white.xml')
    cfg.xmlMedianFile           = osp.join(cfg.xmlDir       , 'main_median.xml')
    cfg.xmlInvRenderFile        = osp.join(cfg.xmlDir       , 'main_invrender.xml')
    cfg.noLightXmlFile          = osp.join(cfg.xmlDir       , 'main_noLight.xml')
    cfg.areaLightXmlFile        = osp.join(cfg.xmlDir       , 'main_areaLight%d.xml')  # % idx
    cfg.eLightXmlFile           = osp.join(cfg.xmlDir       , 'main_eLight.xml')
    cfg.xmlSingleDir            = osp.join(cfg.preprocessDir, 'xmlSingle')
    cfg.xmlFinalFile            = osp.join(cfg.sceneDir     , 'photoscene.xml')
    cfg.finalRenderDir          = osp.join(cfg.sceneDir     , 'rendering')
    cfg.envFile                 = osp.join(cfg.preprocessDir, 'env.hdr')
    cfg.initPerPixelImListFile  = osp.join(cfg.preprocessDir, 'initPerPixelImList.txt')
    cfg.alignedDirByName         = osp.join(cfg.preprocessDir, 'aligned', '%s')  # material name
    cfg.photoByNameId           = osp.join(cfg.alignedDirByName, '%d.jpg')  # material name + view idx
    cfg.maskPhotoByNameId       = osp.join(cfg.alignedDirByName, '%d_maskSN.png')  # material name + view idx
    cfg.maskPhotoJointByNameId  = osp.join(cfg.alignedDirByName, '%d_maskSNJoint.png')  # material name + view idx
    cfg.maskPhotoWeightByNameId = osp.join(cfg.alignedDirByName, '%d_maskSNWeight.png')  # material name + view idx
    cfg.maskGeoByNameId         = osp.join(cfg.alignedDirByName, '%d_maskOR.png')  # material name + view idx
    cfg.maskGeoFromPhotoByNameId        = osp.join(cfg.alignedDirByName, '%d_maskORfromSN.png')
    cfg.maskGeoFromPhotoJointByNameId   = osp.join(cfg.alignedDirByName, '%d_maskORfromSNJoint.png')
    cfg.maskGeoFromPhotoWeightByNameId  = osp.join(cfg.alignedDirByName, '%d_maskORfromSNWeight.png')
    cfg.albedoGeoFromPhotoByNameId      = osp.join(cfg.alignedDirByName, '%d_albedoORfromSN.png')
    cfg.roughGeoFromPhotoByNameId       = osp.join(cfg.alignedDirByName, '%d_roughORfromSN.png')
    cfg.targetGeoFromPhotoByNameId      = osp.join(cfg.alignedDirByName, '%d_targetORfromSN.png')
    cfg.targetBlendGeoFromPhotoByNameId = osp.join(cfg.alignedDirByName, '%d_blendTargetORfromSN.png')
    cfg.uvWarpedByNameId        = osp.join(cfg.alignedDirByName, '%d_imuvcoordWarped.npy')  # material name + view idx
    cfg.uvWarpedVisByNameId     = osp.join(cfg.alignedDirByName, '%d_imuvcoordWarped.png')  # material name + view idx
    cfg.uvUnwarpedVisByNameId   = osp.join(cfg.alignedDirByName, '%d_imuvcoordUnwarped.png')  # material name + view idx
    cfg.warpInfoByNameId        = osp.join(cfg.alignedDirByName, '%d_warpInfo.txt')  # material name + view idx
    cfg.graphParamSaveDirByName = osp.join(cfg.preprocessDir, 'checkpoints', '%s')
    cfg.uvDictPathByName        = osp.join(cfg.graphParamSaveDirByName, 'uvDict.json')
    # Second round (TBA!!!)
    cfg.photoWarpedByNameId         = osp.join(cfg.alignedDirByName, '%d_snColorWarpedExpAdjusted.jpg')
    # idx, similar to cfg.photoOriById

    # Rendering paths
    cfg.renderSingleLightDir    = osp.join(cfg.preprocessDir, 'renderSingleLight')
    cfg.imageSLRenderDirByLight = osp.join(cfg.renderSingleLightDir, '%s', cfg.imageItem)
    cfg.imageSLPathByLightId    = osp.join(cfg.imageSLRenderDirByLight,
        cfg.renderDict[cfg.imageItem]['fn'] + '_%d.' + cfg.renderDict[cfg.imageItem]['ext'])
    cfg.renderCombinedLightDir  = osp.join(cfg.preprocessDir, 'renderCombinedLight')
    cfg.globalEnvmapRenderDir   = osp.join(cfg.renderCombinedLightDir, cfg.envmapItem)
    cfg.globalEnvmapPathById    = osp.join(cfg.globalEnvmapRenderDir,
        cfg.renderDict[cfg.envmapItem]['fn'] + '_%d.' + cfg.renderDict[cfg.envmapItem]['ext'])

    cfg.matWhiteDirByName       = osp.join(cfg.materialDir  , '%s', 'white')  # material name
    cfg.matMedianDirByName      = osp.join(cfg.materialDir  , '%s', 'median')  # material name
    cfg.matInvRenderDirByName   = osp.join(cfg.materialDir  , '%s', 'invrender')  # material name
    cfg.matPhotoSceneInitDirByName   = osp.join(cfg.materialDir  , '%s', 'photoscene_init')  # material name
    cfg.matPhotoSceneFinalDirByName  = osp.join(cfg.materialDir  , '%s', 'photoscene_final')  # material name
    cfg.program                 = osp.join(cfg.rendererRoot , 'OptixRenderer/src/bin/optixRenderer')
    cfg.programMatPart          = osp.join(cfg.rendererRoot , 'OptixRenderer_MatPart/src/bin/optixRenderer')
    cfg.programUV               = osp.join(cfg.rendererRoot , 'OptixRenderer_UVcoord/src/bin/optixRenderer')
    cfg.programLight            = osp.join(cfg.rendererRoot , 'OptixRenderer_Light/src/bin/optixRenderer')
    cfg.graphListFile           = osp.join('txt', 'graphList.txt')
    if cfg.dataMode == 'total3d':
        cfg.total3dInputDir  = osp.join(cfg.inputDir, 'inputs')
        cfg.total3dInputFile = osp.join(cfg.inputDir, 'inputs', '%s.pkl' % cfg.sceneId)
        cfg.total3dOutputDir = osp.join(cfg.inputDir, 'outputs', str(cfg.sceneId))  # total 3d preprocessed result
        cfg.srcCamFile       = osp.join(cfg.preprocessDir, 'cam_src.txt')
        cfg.panopticSrcDir   = osp.join(cfg.preprocessDir, 'panoptic')
        cfg.meshSrcSaveDir   = osp.join(cfg.preprocessDir, 'meshesSrc')  # Input
        cfg.photoSrcDir      = osp.join(cfg.preprocessDir, 'photoSrc')
        cfg.photoSavePath    = osp.join(cfg.photoSrcDir, '0.png')  # Invrender net assumes .png
        cfg.semLabById       = osp.join(cfg.panopticSrcDir, 'semantic', '%d.png')  # idx
        cfg.insLabById          = osp.join(cfg.panopticSrcDir, 'instance', '%d.png')  # idx
        cfg.labelMapFile     = osp.join('txt', 'sunrgbd2ade20kAdd1.txt')
        cfg.gtMaskDir        = None
        cfg.xmlSrcFile       = osp.join(cfg.xmlDir     , 'main_src.xml')

    if cfg.gtMaskDir is not None:  # use provided part masks
        assert (osp.exists(cfg.gtMaskDir))
        cfg.gtMaskByNameId = osp.join(cfg.gtMaskDir, '%s', '%d.png')  # material namd + view idx

    # Command prefix
    cfg.total3dPreprocessCmdPre = 'python3 utils/process_total3d.py --input_pkl %s'  # cfg.total3dInputFile
    cfg.total3dRunCmdPre \
        = 'cd %s; mkdir -p %s; ' % (cfg.total3dRoot, osp.join('external', 'pyTorchChamferDistance', 'build')) \
        + 'python3 main.py configs/total3d.yaml --mode demo --demo_path %s ' \
        + '> %s; ' % osp.join(cfg.logDir, 'total3D.txt') \
        + 'cd %s' % cfg.repoDir  # total 3d dir, data dir, repo
    cfg.blenderApplyUvCmdPre = 'blender -b -P photoscene/applyUV.py -- %s %s > tmp.txt; rm tmp.txt'
    # inputPath, outputPath
    cfg.maskFormerCmdPre \
        = 'cd %s; python3 -W ignore demo/demo_photoscene.py ' % cfg.maskformerRoot \
        + '--config-file configs/ade20k-150-panoptic/maskformer_panoptic_R101_bs16_720k.yaml ' \
        + '--seed %d ' % cfg.seed \
        + '--input %s --output %s --opts MODEL.WEIGHTS demo/model_final_0b3cc8.pkl ' \
        + '> %s; ' % osp.join(cfg.logDir, 'maskFormer.txt') \
        + 'cd %s' % cfg.repoDir
    # renderCmdPre (xmlFile, camFile, outFile )
    cfg.renderPartIdCmdPre  = cfg.programMatPart    + ' -f %s -c %s -o %s -m 7 --modelIdFile ' + cfg.modelIdFile \
                                + ' --seed %d' % cfg.seed \
                                + ' > %s' % osp.join(cfg.logDir, 'renderPartId.txt')
    cfg.renderImageCmdPre   = cfg.program           + ' -f %s -c %s -o %s -m 0 --maxIteration 4' \
                                + ' --seed %d' % cfg.seed \
                                + ' > %s' % osp.join(cfg.logDir, 'renderImage.txt')
    cfg.renderAlbedoCmdPre  = cfg.program           + ' -f %s -c %s -o %s -m 1' \
                                + ' --seed %d' % cfg.seed \
                                + ' > %s' % osp.join(cfg.logDir, 'renderAlbedo.txt')
    cfg.renderNormalCmdPre  = cfg.program           + ' -f %s -c %s -o %s -m 2' \
                                + ' --seed %d' % cfg.seed \
                                + ' > %s' % osp.join(cfg.logDir, 'renderNormal.txt')
    cfg.renderRoughCmdPre   = cfg.program           + ' -f %s -c %s -o %s -m 3' \
                                + ' --seed %d' % cfg.seed \
                                + ' > %s' % osp.join(cfg.logDir, 'renderNormal.txt')
    cfg.renderDepthCmdPre   = cfg.program           + ' -f %s -c %s -o %s -m 5' \
                                + ' --seed %d' % cfg.seed \
                                + ' > %s' % osp.join(cfg.logDir, 'renderDepth.txt')
    cfg.renderUvCmdPre      = cfg.programUV         + ' -f %s -c %s -o %s -m 7' \
                                + ' --seed %d' % cfg.seed \
                                + ' > %s' % osp.join(cfg.logDir, 'renderUV.txt')
    cfg.renderLightCmdPre   = cfg.programLight      + ' -f %s -c %s -o %s -m 7' \
                                + ' --seed %d' % cfg.seed \
                                + ' > %s' % osp.join(cfg.logDir, 'renderLight.txt')  # default sampleNum is 25600
    cfg.invrenderCmdPre         \
        = 'cd %s;' % cfg.invrenderRoot \
        + ' python3 -W ignore testReal.py --cuda --dataRoot %s' \
        + ' --imList %s' \
        + ' --testRoot %s --isLight --isBS --level 2' \
        + ' --seed %d' % cfg.seed \
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
    cfg.renderPartIdInitCmd = cfg.renderPartIdCmdPre % \
        (cfg.xmlFile, cfg.allCamFile, osp.join(cfg.partIdRenderDir, 'im.hdr'))
    cfg.renderNormalInitCmd = cfg.renderNormalCmdPre % \
        (cfg.xmlFile, cfg.allCamFile, osp.join(cfg.normalRenderDir, 'im.hdr'))
    cfg.renderUvInitCmd     = cfg.renderUvCmdPre % \
        (cfg.xmlFile, cfg.allCamFile, osp.join(cfg.uvRenderDir, 'im.hdr'))
    cfg.renderDepthInitCmd  = cfg.renderDepthCmdPre % \
        (cfg.xmlFile, cfg.allCamFile, osp.join(cfg.depthRenderDir, 'im.hdr'))
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
    if not (osp.isfile(imName)):
        print(imName)
        assert (False)
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
            decode_char * channels * height * width, dBuffer), dtype=dtype)  # 0.02 ms ~ 0.1 ms, slow
        depth = depth.reshape([height, width, channels])

        if if_resize:
            if dtype == np.float32:
                depth = cv2.resize(
                    depth, (imWidth, imHeight), interpolation=cv2.INTER_AREA)
            elif dtype == np.int32:
                depth = cv2.resize(
                    depth, (imWidth, imHeight), interpolation=cv2.INTER_NEAREST)

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

    with open(viewDictFile, 'r') as f:
        for line in f.readlines():
            mat, vId = line.strip().split(' ')
            if mat is not None and vId != '-1':
                matDict[mat] = {'vId': int(vId)}

    with open(graphDictFile, 'r') as f:
        for line in f.readlines():
            mat, graph = line.strip().split(' ')
            if mat is not None and mat in matDict.keys():
                matDict[mat]['graph'] = graph
    return matDict


def readViewList(sampledViewFile):
    with open(sampledViewFile, 'r') as f:
        candidateCamIdList = f.readline().strip().split(' ')
        candidateCamIdList = [int(cId) for cId in candidateCamIdList]
    return candidateCamIdList


def getObjMatVertFaceDict(objPath):
    v  = []
    vn = []
    vt = []
    matFacesListDict = {}  # {matName: list of faces}
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
                                'v3': int(v3), 'vt3': int(vt3)}
                elif nItems == 3:
                    v1, vt1, vn1 = vals[1].split('/')
                    v2, vt2, vn2 = vals[2].split('/')
                    v3, vt3, vn3 = vals[3].split('/')

                    if len(vt1) > 0 and len(vt2) > 0 and len(vt3) > 0:
                        faceDict = {'v1': int(v1), 'vt1': int(vt1), 'vn1': int(vn1),
                                    'v2': int(v2), 'vt2': int(vt2), 'vn2': int(vn2),
                                    'v3': int(v3), 'vt3': int(vt3), 'vn3': int(vn3)}
                    else:
                        faceDict = {'v1': int(v1), 'vn1': int(vn1),
                                    'v2': int(v2), 'vn2': int(vn2),
                                    'v3': int(v3), 'vn3': int(vn3)}
                else:
                    print(objPath)
                    print(vals)
                    assert (False)

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
    # Save new mesh #
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


def checkImgListExist(imgList):
    for imgPath in imgList:
        if not osp.exists(imgPath):
            # print('%s not exist!' % imgPath)
            return False
    return True


def readNormal(normalFile, refW, refH):
    ext = normalFile.split('.')[-1]
    if ext == 'npy':
        normalOri = cv2.resize(np.load(normalFile), (refW, refH), interpolation=cv2.INTER_AREA)
    else:
        normalOriImg  = Image.open(normalFile).convert('RGB').resize((refW, refH), Image.ANTIALIAS)
        normalOriImg  = np.asarray(normalOriImg) / 255.0
        normalOri  = normalOriImg * 2 - 1
    normalOri  = normalOri / np.sqrt(np.sum(normalOri ** 2, axis=2))[:, :, np.newaxis]
    normalOri = th.from_numpy(normalOri.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    return normalOri


def loadInvRenderData(cfg, vId, refH, refW, isLoadEnv=False, device='cuda'):
    # Used in preprocess.py and selectGraphFromCls.py
    imFile      = cfg.photoOriById % vId
    imOri       = Image.open(imFile).convert('RGB')
    imOri_resize = np.asarray(imOri.resize((refW, refH), Image.ANTIALIAS), dtype=np.float32) / 255.0
    imOri       = (np.asarray(imOri, dtype=np.float32) / 255.0)
    imOri       = th.from_numpy(imOri).permute(2, 0, 1).unsqueeze(0).to(device)
    target_ref  = th.from_numpy(imOri_resize).permute(2, 0, 1).unsqueeze(0).to(device)  # 1 x 3 x H x W

    normalFile  = cfg.invRenderNormalById % vId
    normalOri   = readNormal(normalFile, refW, refH).to(device)

    albedoFile  = cfg.invRenderAlbedoById % vId
    albedoOri   = Image.open(albedoFile).convert('RGB').resize((refW, refH), Image.ANTIALIAS)
    albedoOri   = (np.asarray(albedoOri) / 255.0) ** 2.2
    albedoOri   = th.from_numpy(albedoOri).permute(2, 0, 1).unsqueeze(0)

    roughFile   = cfg.invRenderRoughById % vId
    roughOri    = Image.open(roughFile).resize((refW, refH), Image.ANTIALIAS)
    roughOri    = (np.asarray(roughOri) / 255.0)
    roughOri    = th.from_numpy(roughOri[np.newaxis, np.newaxis, :, :])

    invRenderData = {'imOri': imOri,
                    'im'    : target_ref,
                    'normal': normalOri,
                    'albedo': albedoOri,
                    'rough': roughOri}

    if isLoadEnv:
        envFile = cfg.invRenderEnvmapById % vId
        envmap     = np.transpose(np.load(envFile)['env'], (4, 0, 1, 2, 3))
        # [bn=1, 3, envRow, envCol, self.envHeight, self.envWidth]
        envmap     = th.from_numpy(np.asarray(envmap)).unsqueeze(0).to(device)  # [1, 3, 119, 160, 8, 16]
        invRenderData['envmap'] = envmap

    return invRenderData


def checkVtExist(objPath):
    if not osp.exists(objPath):
        assert (False)
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
    # oldMatName = 'n/a'
    with open(objPathTemp, 'w') as fw:
        with open(objPath, 'r') as f:
            for line in f.readlines():
                if line.strip().split(' ')[0] == 'usemtl':
                    # oldMatName = line.strip().split(' ')[1]
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


def loadOptInput(cfg, mat, vId, refH, refW, mode='photo', device='cuda'):
    # Default: Predictions from Photos
    # if mode is photo, return the maps from geometry to photo or for photo
    # if mode is geo  , return the maps from photo to geometry or for geometry
    # normal, envmap, uvInput, lossMask, lossWeightMap, targetImg
    assert (mode in ['photo', 'geo'])

    # Load target image
    imFile = cfg.photoOriById % vId if mode == 'photo' else \
                cfg.photoWarpedByNameId % (mat, vId)
    imOri      = Image.open(imFile).convert('RGB')
    imOri_resize = np.asarray(imOri.resize((refW, refH), Image.ANTIALIAS), dtype=np.float32) / 255.0
    imOri      = (np.asarray(imOri, dtype=np.float32) / 255.0)
    imOri = th.from_numpy(imOri).permute(2, 0, 1).unsqueeze(0).to(device)
    target_ref = th.from_numpy(imOri_resize).permute(2, 0, 1).unsqueeze(0).to(device)  # 1 x 3 x H x W

    # Load normal
    normalFile = cfg.invRenderNormalById % vId if mode == 'photo' else \
                    cfg.normalPathById % vId
    normal = readNormal(normalFile, refW, refH).to(device)

    # Load environment maps
    if mode == 'photo':
        envFile = cfg.invRenderEnvmapById % vId
        env     = np.transpose(np.load(envFile)['env'], (4, 0, 1, 2, 3))
        # [bn=1, 3, envRow, envCol, self.envHeight, self.envWidth]
        env     = th.from_numpy(np.asarray(env)).unsqueeze(0).to(device)  # [1, 3, 119, 160, 8, 16]
    else:
        envFile = cfg.globalEnvmapPathById % vId
        env = cv2.imread(envFile, -1)
        # print(env.shape)
        if env.shape[0] == 120 * 8 and env.shape[1] == 160 * 16:
            env = env.reshape(120, 8, 160, 16, 3)
            env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3]))  # 3 x 120 x 160 x 8 x 16
            env = th.from_numpy(env).unsqueeze(0).to(device)  # [1, 3, 120, 160, 8, 16]
        elif env.shape[0] == 120 * 16 and env.shape[1] == 160 * 32:
            env = env.reshape(120, 16, 160, 32, 3)
            env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3]))  # 3 x 120 x 160 x 16 x 32
            env = th.from_numpy(env).view(1, 3 * 120 * 160, 16, 32)
            env = th.nn.functional.interpolate(env, size=(8, 16), mode='bilinear', align_corners=True)
            env = env.view(1, 3, 120, 160, 8, 16)
        else:
            assert (False)

    # Load UV coord maps
    if mode == 'photo':
        uvMapWarpedPath = cfg.uvWarpedByNameId % (mat, vId)
        uvMap = th.from_numpy(np.load(uvMapWarpedPath)).to(device)
        uvMap = th.nn.functional.interpolate(uvMap.permute(0, 3, 1, 2), size=(refH, refW)).permute(0, 2, 3, 1)
    else:
        uvPath = cfg.uvPathById % (vId)
        uvMap = loadBinary(uvPath, channels=2, if_resize=True, imWidth=refW, imHeight=refH)  # 1 x refH x refW x 2
        uvMap = th.from_numpy(uvMap).to(device)

    # Load loss mask and loss weight map
    maskSNPath = cfg.maskPhotoByNameId % (mat, vId) if mode == 'photo' else \
                    cfg.maskGeoFromPhotoByNameId % (mat, vId)
    lossMask = th.from_numpy(np.asarray(
        Image.open(maskSNPath).resize((refW, refH))).astype(float) / 255).unsqueeze(0).unsqueeze(1).to(device)

    # Load weighted loss mask
    maskSNWeightPath = cfg.maskPhotoWeightByNameId % (mat, vId) if mode == 'photo' else \
                        cfg.maskGeoFromPhotoWeightByNameId % (mat, vId)
    lossWeightMap = th.from_numpy(np.asarray(
        Image.open(maskSNWeightPath).resize((refW, refH))).astype(float) / 255).unsqueeze(0).unsqueeze(1).to(device)

    # output dict
    invRenderData = {'imOri': imOri,
                    'im'    : target_ref,
                    'normal': normal,
                    'envmap': env,
                    'uvInput': uvMap,
                    'lossMask': lossMask.long(),
                    'lossWeightMap': lossWeightMap}

    return invRenderData


def getMaterialSavePathDict(cfg, mode, ext='png'):
    assert (mode in ['init', 'final'])
    matPhotoSceneDirByName \
        = cfg.matPhotoSceneInitDirByName if mode == 'init' else cfg.matPhotoSceneFinalDirByName
    selectedGraphDict = readGraphDict(cfg.selectedGraphFile)
    matSavePathDict = {}
    for partName, _ in selectedGraphDict.items():
        matOptDir = matPhotoSceneDirByName % partName
        fileList = []
        for matParam in ['basecolor', 'normal', 'roughness']:
            fullPath = osp.join(matOptDir, '%s.%s' % (matParam, ext))
            if not os.path.exists(fullPath):
                print('Warning! %s not exist!' % fullPath)
                assert (False)
            fileList.append(fullPath)
        matSavePathDict[partName] = fileList
    return matSavePathDict
