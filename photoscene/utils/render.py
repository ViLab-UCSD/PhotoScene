import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image

from utils.io import checkImgListExist, loadBinary


def hdr2ldr(rgbeName, ldrName, noNormalize=False):
    im = cv2.imread(rgbeName, -1)
    if not noNormalize:
        imMean = np.mean(im)
        im = im / imMean * 0.5
    im = np.clip(im, 0, 1)
    im = im ** (1.0 / 2.2)
    im = (255 * im).astype(np.uint8)
    cv2.imwrite(ldrName, im)
    print('Converted to LDR image at %s!' % ldrName)


def renderPass(cfg, renderDir, passName, xmlFile, camFile, camIdList,
                    isOverwrite=False, noNormalize=False):
    renderDict = cfg.renderDict
    fn, ext = renderDict[passName]['fn'], renderDict[passName]['ext']

    outDir = osp.join(renderDir, passName)
    if not osp.exists(outDir):
        os.system('mkdir -p %s' % outDir)
    outFile = osp.join(outDir, 'im.hdr')

    outList = [osp.join(outDir, '%s_%d.%s' % (fn, int(vId), ext)) for vId in camIdList]
    if checkImgListExist(outList) and not isOverwrite:
        print('Images in %s are rendered! Pass!' % outDir)
    else:
        if passName == 'image':
            cmd = cfg.renderImageCmdPre % (xmlFile, camFile, outFile)
        elif passName == 'albedo':
            cmd = cfg.renderAlbedoCmdPre % (xmlFile, camFile, outFile)
        elif passName == 'normal':
            cmd = cfg.renderNormalCmdPre % (xmlFile, camFile, outFile)
        elif passName == 'rough':
            cmd = cfg.renderRoughCmdPre % (xmlFile, camFile, outFile)
        elif passName == 'depth':
            cmd = cfg.renderDepthCmdPre % (xmlFile, camFile, outFile)
        elif passName == 'light':
            cmd = cfg.renderLightCmdPre % (xmlFile, camFile, outFile)
        elif passName == 'uvCoord':
            cmd = cfg.renderUvCmdPre % (xmlFile, camFile, outFile)
        elif passName == 'partId':
            cmd = cfg.renderPartIdCmdPre % (xmlFile, camFile, outFile)
        else:
            assert (False)

        print('Rendering %s at %s ... ' % (passName, outDir), end='')
        print('(This may take a while)') if passName in ['image', 'light'] else print('')
        os.system(cmd)

        with open(camFile, 'r') as f:
            for line in f.readlines():
                cNum = int(line.strip())
                break
        # print('Total valid views: %d' % cNum)

        # Adjust View ID
        tmpDir = outDir.replace(passName, '%s_tmp' % passName)
        os.system('mkdir -p %s' % tmpDir)
        for j in range(cNum):
            i = cNum - j - 1
            old = osp.join(outDir, '%s_%d.%s' % (fn, i + 1, ext))
            assert (osp.exists(old))
            new = osp.join(tmpDir, '%s_%s.%s' % (fn, camIdList[i], ext))
            cmd = 'mv %s %s' % (old, new)
            # print(cmd)
            os.system( cmd)
        assert (osp.exists(tmpDir))
        os.system('rm -r %s' % outDir)
        os.system('mv %s %s' % (tmpDir, outDir))
        # print('Finish rendering locally!')

        if passName == 'image':
            # save png format
            for selectedId in camIdList:
                rgbeName = osp.join(outDir, '%s_%s.rgbe' % (fn, selectedId))
                ldrName = rgbeName.replace('.rgbe', '.png')
                hdr2ldr(rgbeName, ldrName, noNormalize=noNormalize)
            # print('Converted HDR to LDR!')

        if passName == 'uvCoord':
            # save visualization format
            for selectedId in camIdList:
                uvPath = osp.join(outDir, '%s_%s.dat' % (fn, selectedId))
                uvMap = loadBinary(uvPath, channels=2)[0, :]  # 1 x refH x refW x 2 -> refH x refW x 2
                im = uvMap - np.floor(uvMap)
                im = Image.fromarray(
                    (np.concatenate([im, np.ones((im.shape[0], im.shape[1], 1))], axis=2) * 255.0).astype(np.uint8))
                im.save(os.path.join(uvPath.replace('.dat', '_vis.jpg')))

        if passName == 'light':
            for selectedId in camIdList:
                envFile = osp.join(outDir, '%s_%d.hdr' % (fn, selectedId))
                envmap = cv2.imread(envFile, -1)
                if envmap.shape[0] == 120 * 8 and envmap.shape[1] == 160 * 16:
                    envHeight, envWidth = 8, 16
                elif envmap.shape[0] == 120 * 16 and envmap.shape[1] == 160 * 32:
                    envHeight, envWidth = 16, 32
                else:
                    assert (False)
                envmap = envmap.reshape(120, envHeight, 160, envWidth, 3)
                envmap = np.ascontiguousarray(envmap.transpose([4, 0, 2, 1, 3]))  # 3 x 120 x 160 x envHeight x envWidth

                envmap = np.transpose(envmap, [1, 2, 3, 4, 0])  # 120 x 160 x envHeight x envWidth x 3
                nrows, ncols, gap = 24, 16, 1
                envRow, envCol = envmap.shape[0], envmap.shape[1]

                interY = int(envRow / nrows)
                interX = int(envCol / ncols)

                lnrows = len(np.arange(0, envRow, interY))
                lncols = len(np.arange(0, envCol, interX))

                lenvHeight = lnrows * (envHeight + gap) + gap
                lenvWidth = lncols * (envWidth + gap) + gap

                envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
                for r in range(0, envRow, interY):
                    for c in range(0, envCol, interX):
                        rId = int(r / interY)
                        cId = int(c / interX)

                        rs = rId * (envHeight + gap)
                        cs = cId * (envWidth + gap)
                        envmapLarge[rs : rs + envHeight, cs : cs + envWidth, :] = envmap[r, c, :, :, :]

                envmapLarge = np.clip(envmapLarge, 0, 1)
                envmapLarge = (255 * (envmapLarge ** (1.0 / 2.2))).astype(np.uint8)
                cv2.imwrite(envFile.replace('.hdr', '_vis.jpg'), envmapLarge[:, :, ::-1])
