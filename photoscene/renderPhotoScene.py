import argparse

from utils.io import *
from utils.xml import *
from utils.render import *

if __name__ == '__main__':
    print('\n')

    parser = argparse.ArgumentParser(description='Script for Rendering Final PhotoScene Result')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    xmlBaseFile = cfg.xmlMedianFile
    objSaveDir = cfg.meshSaveFinalDir
    if not osp.exists(objSaveDir):
        os.system('mkdir -p %s' % objSaveDir)
    xmlSingleDir = cfg.xmlSingleDir
    renderDir = cfg.finalRenderDir
    sampledCamFile = cfg.sampledCamFile
    sampledViewFile = cfg.sampledCamIdFile # sampledViewList
    candidateCamIdList = readViewList(sampledViewFile)
    matSavePathDict = getMaterialSavePathDict(cfg, mode='final')

    ##### Generate new xml file, apply uv to save new obj, update material paths
    objInfoDict = getObjInfoDict(xmlBaseFile)
    # Collect material UV dicts and apply to OBJ
    objMatUVDictPath = osp.join(objSaveDir, 'objMatUVDict.json')
    # warpInfoDir = osp.join(orSnViewDir, 'output')
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

    ##### Update objDict to each light xml file #####
    xmlCombinedFile = osp.join(xmlSingleDir, 'main_combined.xml')
    objInfoDictCombined = getObjInfoDict(xmlCombinedFile)
    for objID, objInfo in objInfoDictCombined.items():
        shape_ID = objInfo['shape_id']
        if shape_ID in objInfoDictNewUV2.keys() and not objInfo['isContainer'] and not objInfo['isEmitter']:
            objInfoDictCombined[objID] = objInfoDictNewUV2[shape_ID]

    xmlFileNew = cfg.xmlFinalFile
    xmlFileNew = updateXmlNewMatAndShapePath(xmlCombinedFile, xmlFileNew, matSavePathDict, \
            objInfoDictCombined, nSample=cfg.nSamplePerPixel)

    renderPass(cfg, renderDir, 'image', xmlFileNew, sampledCamFile, \
            candidateCamIdList, noNormalize=True)
