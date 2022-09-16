import os
import argparse
import json
from utils.io import load_cfg, readGraphDict, readViewDict, loadOptInput
from utils.xml import getSensorInfoDict
from utils.rendering_layer import MicrofacetUV
import time
import math
import random
import torch as th
import numpy as np

from diffmat import MaterialGraphTranslator as MGT
from diffmat.optim import Optimizer
from diffmat.optim import ParamSampler
from diffmat.core.io import write_image
from pathlib import Path
from typing import Dict, Optional, Any


class LossObject(object):
    def __init__(self, renderObj, optObjectiveDict, targetImg, device='cuda'):
        self.renderObj  = renderObj
        self.targetImg  = targetImg
        self.optObjectiveDict = optObjectiveDict
        self.device     = device
        self.err        = math.inf
        self.err_stat   = math.inf
        self.err_pix    = math.inf
        self.err_vgg    = math.inf
        self.err_style  = math.inf

    def compute(self, renderImg, output_dict=False):
        self.err_stat = self.renderObj.applyMaskStatLoss(renderImg, self.targetImg)
        coeff_stat = self.optObjectiveDict['stat'] if 'stat' in self.optObjectiveDict.keys() else \
            th.tensor([0]).to(self.device)

        self.err_pix = self.renderObj.applyMaskLoss(renderImg, self.targetImg)
        coeff_pix = self.optObjectiveDict['pix'] if 'pix' in self.optObjectiveDict.keys() else \
            th.tensor([0]).to(self.device)

        self.err_vgg = self.renderObj.applyMaskVggLoss(renderImg, self.targetImg)
        coeff_vgg = self.optObjectiveDict['vgg'] if 'vgg' in self.optObjectiveDict.keys() else \
            th.tensor([0]).to(self.device)

        self.err_style = self.renderObj.applyMaskStyleLoss(renderImg, self.targetImg)
        coeff_style = self.optObjectiveDict['style'] if 'style' in self.optObjectiveDict.keys() else \
            th.tensor([0]).to(self.device)

        self.err = self.err_stat * coeff_stat + self.err_pix * coeff_pix + self.err_vgg * coeff_vgg + \
            self.err_style * coeff_style
        err_dict = {'stat': self.err_stat, 'pix': self.err_pix, 'vgg': self.err_vgg, 'style': self.err_style}

        if output_dict:
            return self.err, err_dict
        else:
            return self.err

    def print(self, iter_numer=None, iter_denom=None, iter_mode=None, add_text=None):
        if iter_numer is not None and iter_denom is not None:
            d_str = str(int(iter_denom))
            n_str = str(int(iter_numer)).rjust(len(d_str))
            msg = '[%s/%s] ' % (n_str, d_str)
        elif iter_numer is not None:
            msg = '[%d] ' % (int(iter_numer))
        else:
            msg = ''
        if iter_mode is not None:
            msg = msg.replace('[', '[%s ' % iter_mode)
        msg += 'total: %.4f; ' % self.err.cpu().detach().item()
        if 'stat' in self.optObjectiveDict.keys():
            msg += 'stat: %.4f; ' % self.err_stat.cpu().detach().item()
        if 'pix' in self.optObjectiveDict.keys():
            msg += 'pix: %.4f; ' % self.err_pix.cpu().detach().item()
        if 'vgg' in self.optObjectiveDict.keys():
            msg += 'vgg: %.4f; ' % self.err_vgg.cpu().detach().item()
        if 'style' in self.optObjectiveDict.keys():
            msg += 'style: %.4f; ' % self.err_style.cpu().detach().item()
        if add_text is not None:
            msg += '%s' % add_text

        print(msg)


class HomogeneousOptimizer(object):
    def __init__(self, renderObj, optObjectiveDict, targetImg, paramSaveDir, matSaveDir, mode,
                img_format='png', texRes=16, input_seed=0, device='cuda'):
        self.renderObj          = renderObj
        self.optObjectiveDict   = optObjectiveDict
        self.targetImg          = targetImg
        self.device             = device
        self.lossObj            = LossObject(renderObj, optObjectiveDict, targetImg, device=self.device)
        if input_seed is not None:
            th.manual_seed(input_seed)
            random.seed(input_seed)
            np.random.seed(input_seed)

        self.paramSaveDir       = Path(paramSaveDir)
        self.paramInitSavePath  = Path(paramSaveDir) / 'init.pth'
        self.paramOptSavePath   = Path(paramSaveDir) / 'latest.pth'
        self.paramFinalSavePath = Path(paramSaveDir) / Path('%s.pth' % mode)
        self.uvDictSavePath     = Path(paramSaveDir) / 'uvDict.json'
        self.errorSavePath      = Path(paramSaveDir) / 'error.txt'
        self.matSaveDir         = matSaveDir

        self.normal = th.tensor([0.5, 0.5, 1]).to(self.device)
        self.albedo = th.tensor([1.0, 1.0, 1.0]).to(self.device)
        self.optimizerAlbedo = th.optim.Adam([self.albedo.requires_grad_()], lr=1e-3)
        self.rough = th.tensor([0.5]).to(self.device)
        self.optimizerRough  = th.optim.Adam([self.rough.requires_grad_()], lr=5e-3)
        self.input_seed = input_seed
        self.eps = 1e-10
        self.res = texRes  # For output texture map resolution
        self.img_format = img_format

        self.loss_min = math.inf
        self.param_alb_min = self.albedo
        self.param_rou_min = self.rough

    def _update_param(self, param_dict):
        if 'albedo' in param_dict.keys():
            self.albedo = th.tensor(param_dict['albedo']).to(self.device)
            self.optimizerAlbedo = th.optim.Adam([self.albedo.requires_grad_()], lr=1e-3)
        if 'rough' in param_dict.keys():
            self.rough = th.tensor(param_dict['rough']).to(self.device)
            self.optimizerRough  = th.optim.Adam([self.rough.requires_grad_()], lr=5e-3)

    def _save_state(self, iter_num, save_file):
        # Similar state saving format as MATch
        state = {
            'iter': iter_num,
            'param_alb': self.albedo.cpu().detach().numpy(),
            'optim_alb': self.optimizerAlbedo.state_dict(),
            'param_rou': self.rough.cpu().detach().numpy(),
            'optim_rou': self.optimizerRough.state_dict(),
            'loss_min': self.loss_min,
            'param_alb_min': self.param_alb_min.cpu().detach().numpy(),
            'param_rou_min': self.param_rou_min.cpu().detach().numpy()
        }
        th.save(state, save_file)

    def _load_state(self, save_file):
        # Similar state loading format as MATch
        state: Dict[str, Any] = th.load(save_file)
        self._update_param({'albedo': state['param_alb'], 'rough': state['param_rou']})
        self.optimizerAlbedo.load_state_dict(state['optim_alb'])
        self.optimizerRough.load_state_dict(state['optim_rou'])
        self.loss_min: float = state['loss_min']
        self.param_alb_min: th.Tensor = th.tensor(state['param_alb_min'])
        self.param_rou_min: th.Tensor = th.tensor(state['param_rou_min'])

        return state['iter']

    def save_images(self, map_dict):
        for header, img in map_dict.items():
            write_image(img.squeeze(0), (self.matSaveDir / header), self.img_format)

    def load_param(self, resumeMode):
        assert (resumeMode in ['first', 'init', 'latest', 'second'])
        self._load_state(Path(paramSaveDir) / Path('%s.pth' % resumeMode))

    def eval_homo(self, output_maps=False, detach=False):

        map_dict = {'basecolor': self.albedo.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.res, self.res),
                    'normal': self.normal.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.res, self.res),
                    'roughness': self.rough.repeat(1, 1, self.res, self.res)}

        map_dict['basecolor'].clamp_(min=self.eps, max=1)
        map_dict['roughness'].clamp_(min=0.1, max=1)

        if detach:
            for k, v in map_dict.items():
                map_dict[k] = v.detach()

        normal, albedo, rough \
                        = (map_dict['normal'] - 0.5) * 2.0, map_dict['basecolor'] ** 2.2, map_dict['roughness']

        rendered_img = self.renderObj.eval(albedo, normal, rough)  # clamp to (eps, 1), ** (1/2.2)

        if output_maps:
            return rendered_img, map_dict
        else:
            return rendered_img

    def init_param_search(self, sample_num=20, save_params=True):
        print('\nSearching for initial parameters ...')
        # Look for better start point
        bestErr = 10000000000
        for sample_id in range(sample_num):
            albedo = th.rand(3)
            self._update_param({'albedo': albedo})

            sample_rendered_img = self.eval_homo()
            err = self.lossObj.compute(sample_rendered_img)

            if err < bestErr:
                is_better = 'better!'
                bestErr = err.cpu().detach().numpy().item()
                bestAlbedo = albedo
                self.loss_min = bestErr
                self.param_alb_min = bestAlbedo
                self.lossObj.print(iter_numer=sample_id + 1, iter_denom=sample_num, add_text=is_better)
            else:
                is_better = ''

        self._update_param({'albedo': bestAlbedo.cpu().detach().numpy()})
        if save_params:
            self._save_state(iter_num=0, save_file=self.paramInitSavePath)

    def optimize(self, num_iter=1000, save_itv=100):
        print('\nOptimizing for material parameters ...')
        err_list = []
        # log start time
        opt_start = time.time()
        accu = 0
        accu_time = 0
        movingAvgPre = 1000000
        # start optimization
        for i in range(0, num_iter):
            iter_start = time.time()

            self.optimizerAlbedo.zero_grad()
            self.optimizerRough.zero_grad()

            rendered_img = self.eval_homo()

            err = self.lossObj.compute(rendered_img)

            # log error
            err_list.append(err.cpu().detach().numpy().item())

            if err < self.loss_min:
                self.loss_min = err
                self.param_alb_min = self.albedo
                self.param_rou_min = self.rough

            # save intermediate result
            if (i % save_itv) == 0 and i != 0:
                movingAvg = accu / save_itv
                if movingAvg > movingAvgPre or ( (movingAvgPre - movingAvg) < 0.01 * movingAvgPre):
                    print('Early stopping!')
                    break
                else:
                    self._save_state(i, self.paramOptSavePath)

                accu = 0
                movingAvgPre = movingAvg

                time_avg = accu_time / save_itv
                accu_time = 0

                accu_err_text = 'accu_err: %.06f; ' % movingAvg
                time_text = 'time: %d ms/iter' % int(time_avg)
                self.lossObj.print(iter_numer=i, iter_denom=num_iter, add_text=accu_err_text + time_text)

            err.backward()
            accu += err.cpu().detach().numpy().item()

            self.optimizerAlbedo.step()
            self.optimizerRough.step()

            iter_end = time.time()
            accu_time += (iter_end - iter_start) * 1000.0

        # print time cost for the entire optimization
        opt_end = time.time()
        print('Runtime: %f' % (opt_end - opt_start))

        # save final optimization state
        self._save_state(i, self.paramFinalSavePath)

        # save optimized material maps
        _, map_dict = self.eval_homo(output_maps=True, detach=True)
        self.save_images(map_dict)
        print('Optimized materials are saved at %s\n' % self.matSaveDir)

        # save UV optimization results
        uvValDict = {'scaleU': 1.0, 'scaleV': 1.0, 'rot': 0.0, 'transU': 0.0, 'transV': 0.0}
        with open(self.uvDictSavePath, 'w') as fp:
            json.dump(uvValDict, fp, indent=4)

        np.savetxt(self.errorSavePath, np.array(err_list), fmt='%f')


class GraphSampler(ParamSampler):
    def sample_graph_params(self, ):
        graph = self.graph
        level_kwargs = self.level_kwargs
        param_config = {}
        with self.temp_rng(self.rng_state):

            # Get the current parameter setting
            params = graph.get_parameters_as_tensor(**level_kwargs)
            new_params = self.func(params)  # sample new parameters
            graph.set_parameters_from_tensor(new_params, **level_kwargs)
            graph.set_parameters_from_config(param_config)

        return new_params


class GraphOptimizer(Optimizer):
    def __init__(self, graph, renderObj, uvConfigDict, optObjectiveDict, targetImg, paramSaveDir, matSaveDir,
                        mode, img_format='png', fixUv=False, fixGraph=False, useHomoRough=True, input_seed=0):
        super(GraphOptimizer, self).__init__(graph)

        self.renderObj          = renderObj
        self.optObjectiveDict   = optObjectiveDict
        self.targetImg          = targetImg
        self.device             = self.graph.device
        self.lossObj            = LossObject(renderObj, optObjectiveDict, targetImg, device=self.device)
        self.paramSaveDir       = Path(paramSaveDir)
        self.paramInitSavePath  = Path(paramSaveDir) / 'init.pth'
        self.paramOptSavePath   = Path(paramSaveDir) / 'latest.pth'
        self.paramFinalSavePath = Path(paramSaveDir) / Path('%s.pth' % mode)
        self.uvDictSavePath     = Path(paramSaveDir) / 'uvDict.json'
        self.albScaleSavePath   = Path(paramSaveDir) / 'albScale.json'
        self.errorSavePath      = Path(paramSaveDir) / 'error.txt'
        self.matSaveDir         = Path(matSaveDir)

        self.albScale = th.tensor([1.0, 1.0, 1.0]).to(self.device)
        self.optimizerAlbScale = th.optim.Adam([self.albScale.requires_grad_()], lr=1e-3)
        self.useHomoRough = useHomoRough
        if useHomoRough:
            self.roughHomo = th.tensor([0.5]).to(self.device)
            self.optimizerRough  = th.optim.Adam([self.roughHomo.requires_grad_()], lr=1e-3)
        else:
            self.rouScale = th.tensor([1.0]).to(self.device)
            self.optimizerRouScale = th.optim.Adam([self.rouScale.requires_grad_()], lr=1e-3)
        self.input_seed = input_seed
        self.eps = 1e-10
        self.res = 2 ** self.graph.res
        self.img_format = img_format

        algo_kwargs = {'min': -0.05, 'max': 0.05, 'mu': 0.0, 'sigma': 0.03}
        self.sampler = GraphSampler(self.graph, algo='uniform', algo_kwargs=algo_kwargs, seed=input_seed,
                            device=self.device)

        self.fixUv = fixUv
        uvScale = th.tensor([uvConfigDict['uvInitScale'], uvConfigDict['uvInitScale']])
        uvRot = th.tensor([uvConfigDict['uvInitRot']])
        uvTrans = th.tensor([uvConfigDict['uvInitTrans'], uvConfigDict['uvInitTrans']])
        uvScaleLog = th.log(uvScale) / th.log(th.tensor([uvConfigDict['uvScaleBase'], uvConfigDict['uvScaleBase']]))
        self.uvDict = {'scale': uvScale.to(self.device), 'rot': uvRot.to(self.device), 'trans': uvTrans.to(self.device),
                    'scaleBase': th.tensor([uvConfigDict['uvScaleBase'], uvConfigDict['uvScaleBase']]).to(self.device),
                    'logscale': uvScaleLog.to(self.device)}
        self.uvScaleExpMin  = uvConfigDict['uvScaleExpMin']
        self.uvScaleExpMax  = uvConfigDict['uvScaleExpMax']
        self.uvScaleStepNum = uvConfigDict['uvScaleStepNum']
        self.uvRotStepNum   = uvConfigDict['uvRotStepNum']
        self.uvTransStepNum = uvConfigDict['uvTransStepNum']

        self.fixGraph   = fixGraph

    def _save_state(self, iter_num, save_file):
        # Similar state saving format as MATch
        state = {
            'iter': iter_num,
            'param': self.graph.get_parameters_as_tensor().cpu(),
            'optim': self.optimizer.state_dict(),
            'loss_min': self.loss_min,
            'param_min': self.param_min.cpu(),
            'uv': self.uvDict,
            'alb_scale': self.albScale.cpu()
        }
        th.save(state, save_file)

    def _load_state(self, save_file):
        # Similar state loading format as MATch
        state: Dict[str, Any] = th.load(save_file)
        params: Optional[th.Tensor] = state['param']
        if params is not None:
            self.graph.set_parameters_from_tensor(params.to(self.device))
        self.optimizer.load_state_dict(state['optim'])
        self.loss_min: float = state['loss_min']
        self.param_min: th.Tensor = state['param_min']
        self.uvDict = state['uv']
        self.albScale = state['alb_scale'].to(self.device)

        return state['iter']

    def save_images(self, map_dict):
        for header, img in map_dict.items():
            write_image(img.squeeze(0), (self.matSaveDir / header), self.img_format)

    def load_param(self, resumeMode):
        assert (resumeMode in ['first', 'init', 'latest', 'second'])
        self._load_state(Path(paramSaveDir) / Path('%s.pth' % resumeMode))

    def init_param_search(self, sample_num=20, save_params=True):
        print('\nSearching for initial graph parameters ...')
        # Search for optimal
        bestErr = 10000000000
        for sample_idx in range(sample_num):
            sample_params = self.sampler.sample_graph_params()
            sample_rendered_img = self.eval_graph()
            err = self.lossObj.compute(sample_rendered_img)

            if err < bestErr:
                bestErr = err.cpu().detach().numpy().item()
                bestParam = sample_params
                self.loss_min = bestErr
                self.param_min = bestParam
                is_better = 'better!'

                self.lossObj.print(iter_numer=sample_idx + 1, iter_denom=sample_num, add_text=is_better)

        self.graph.set_parameters_from_tensor(bestParam, **self.sampler.level_kwargs)
        if save_params:
            # save parameters
            self._save_state(iter_num=0, save_file=self.paramInitSavePath)

    def eval_graph(self, output_maps=False, detach=False):
        maps = self.graph.evaluate_maps()
        if detach:
            maps = [s.detach() for s in maps]

        map_dict = {}
        for img, header in zip(maps, self.sampler.image_headers):
            if header in ['basecolor', 'normal', 'roughness']:
                if img.shape[1] > 3:
                    img = img[:, :3]
                map_dict[header] = img

        if self.albScale is not None:
            map_dict['basecolor'] *= self.albScale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        map_dict['basecolor'].clamp_(min=self.eps, max=1)

        if not self.useHomoRough:
            map_dict['roughness'] *= self.rouScale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            map_dict['roughness'] = self.roughHomo.repeat(1, 1, self.res, self.res)

        map_dict['roughness'].clamp_(min=0.1, max=1)

        normal, albedo, rough \
                        = (map_dict['normal'] - 0.5) * 2.0, map_dict['basecolor'] ** 2.2, map_dict['roughness']

        rendered_img = self.renderObj.eval(albedo, normal, rough, uvDict=self.uvDict)  # clamp to (eps, 1), ** (1/2.2)

        if output_maps:
            return rendered_img, map_dict
        else:
            return rendered_img

    def searchUVSeq(self):
        # print('Searching for optimal UV parameters sequentially ...')
        angStep = self.uvRotStepNum
        expMin = self.uvScaleExpMin
        expMax = self.uvScaleExpMax
        expStep = self.uvScaleStepNum
        transStep = self.uvTransStepNum

        bestErr = 100000000000
        uvScaleLogInit  = self.uvDict['logscale']
        uvTransInit     = self.uvDict['trans']
        uvScaleBase     = self.uvDict['scaleBase']
        iter = 0
        for ang in range(angStep):
            iter += 1
            uvScaleLog = uvScaleLogInit
            uvScale = th.pow(uvScaleBase, uvScaleLog)
            uvRot = th.tensor([2 * math.pi / angStep * ang])
            uvTrans = uvTransInit
            self.uvDict = {
                'scale': uvScale.to(self.device), 'rot': uvRot.to(self.device), 'trans': uvTrans.to(self.device),
                'scaleBase': uvScaleBase.to(self.device), 'logscale': uvScaleLog.to(self.device)}

            rendered_img = self.eval_graph(detach=True)
            err = self.lossObj.compute(rendered_img)

            if err < bestErr:
                uvScaleBest = uvScale
                uvRotBest = uvRot
                uvTransBest = uvTrans
                uvScaleLogBest = uvScaleLog
                bestErr = err.cpu().detach().numpy().item()

        for scaleIdx in range(expStep):
            scaleExp = (expMax - expMin) / expStep * scaleIdx + expMin
            iter += 1
            uvScaleLog = th.tensor([scaleExp, scaleExp]).to(self.device)
            uvScale = th.pow(uvScaleBase, uvScaleLog)
            uvRot = uvRotBest
            uvTrans = uvTransBest
            self.uvDict = {
                'scale': uvScale.to(self.device), 'rot': uvRot.to(self.device), 'trans': uvTrans.to(self.device),
                'scaleBase': uvScaleBase.to(self.device), 'logscale': uvScaleLog.to(self.device)}

            rendered_img = self.eval_graph(detach=True)
            err = self.lossObj.compute(rendered_img)

            if err < bestErr:
                uvScaleBest = uvScale
                uvRotBest = uvRot
                uvTransBest = uvTrans
                uvScaleLogBest = uvScaleLog
                bestErr = err.cpu().detach().numpy().item()

        for idxX, transX in enumerate(list(np.arange(-0.5, 0.5, 1 / transStep))):
            for idxY, transY in enumerate(list(np.arange(-0.5, 0.5, 1 / transStep))):
                iter += 1
                uvScaleLog = uvScaleLogBest
                uvScale = uvScaleBest
                uvRot = uvRotBest
                uvTrans = th.tensor([float(transX), float(transY)]).to(self.device)
                self.uvDict = {
                    'scale': uvScale.to(self.device), 'rot': uvRot.to(self.device), 'trans': uvTrans.to(self.device),
                    'scaleBase': uvScaleBase.to(self.device), 'logscale': uvScaleLog.to(self.device)}

                rendered_img = self.eval_graph(detach=True)
                err = self.lossObj.compute(rendered_img)

                if err < bestErr:
                    uvScaleBest = uvScale
                    uvRotBest = uvRot
                    uvTransBest = uvTrans
                    uvScaleLogBest = uvScaleLog
                    bestErr = err.cpu().detach().numpy().item()

        # Optimal UV parameters
        self.uvDict = {
            'scale': uvScaleBest.to(self.device), 'rot': uvRotBest.to(self.device),
            'trans': uvTransBest.to(self.device), 'scaleBase': uvScaleBase.to(self.device),
            'logscale': uvScaleLogBest.to(self.device)}

    def optimize(self, num_iter=1000, save_itv=100, search_itv=50):
        print('\nOptimizing for material parameters ...')
        err_list = []

        # log start time
        opt_start = time.time()
        accu = 0
        accu_time = 0
        movingAvgPre = 1000000
        # start optimization
        for i in range(0, num_iter):
            iter_start = time.time()

            isSearchUv = False
            i_material = i
            if i_material % search_itv == 0 and not self.fixUv:
                isSearchUv = True

            if not self.fixGraph:
                self.optimizer.zero_grad()
            self.optimizerAlbScale.zero_grad()
            if self.useHomoRough:
                self.optimizerRough.zero_grad()
                rendered_img = self.eval_graph()
            else:
                self.optimizerRouScale.zero_grad()
                rendered_img = self.eval_graph()

            err, err_dict = self.lossObj.compute(rendered_img, output_dict=True)

            # log error
            err_list.append(err.cpu().detach().numpy().item())

            if err < self.loss_min:
                self.loss_min = err
                self.param_min = self.graph.get_parameters_as_tensor()

            # save intermediate result
            if (i % save_itv) == 0 and i != 0:
                movingAvg = accu / save_itv
                if movingAvg > movingAvgPre or ( (movingAvgPre - movingAvg) < 0.01 * movingAvgPre):
                    print('Early stopping!')
                    break
                else:
                    self._save_state(i, self.paramOptSavePath)

                accu = 0
                movingAvgPre = movingAvg

                time_avg = accu_time / save_itv
                accu_time = 0

                uv_text = 'scale: %.3f %.3f; rot: %.3f; trans: %.3f %.3f; ' % \
                    (self.uvDict['scale'][0].item(), self.uvDict['scale'][1].item(), self.uvDict['rot'].item(),
                    self.uvDict['trans'][0].item(), self.uvDict['trans'][1].item())

                accu_err_text = 'accu: %.06f; ' % movingAvg
                time_text = 'time: %d ms/iter' % int(time_avg)
                self.lossObj.print(iter_numer=i, iter_denom=num_iter, add_text=accu_err_text + uv_text + time_text)

            err.backward()
            accu += err.cpu().detach().numpy().item()

            if not self.fixGraph:
                self.optimizer.step()
            self.optimizerAlbScale.step()

            isOptRough = False
            if i_material > 100 or err_dict['stat'].cpu().detach().numpy().item() < 0.01:
                isOptRough = True
            if isOptRough:
                if self.useHomoRough:
                    self.optimizerRough.step()
                else:
                    self.optimizerRouScale.step()

            iter_end = time.time()
            accu_time += (iter_end - iter_start) * 1000.0

            if isSearchUv:
                self.searchUVSeq()

        # print time cost for the entire optimization
        opt_end = time.time()
        print('Runtime: %.1f' % (opt_end - opt_start))

        # save final parameters
        self._save_state(i, self.paramFinalSavePath)

        # save optimized material maps
        _, map_dict = self.eval_graph(output_maps=True, detach=True)
        self.save_images(map_dict=map_dict)
        print('Optimized materials are saved at %s\n' % self.matSaveDir)

        # save UV optimization results
        uvValDict = {'scaleU': self.uvDict['scale'][0].item(),
                    'scaleV': self.uvDict['scale'][1].item(),
                    'rot': self.uvDict['rot'].item(),
                    'transU': self.uvDict['trans'][0].item(),
                    'transV': self.uvDict['trans'][1].item()}
        with open(self.uvDictSavePath, 'w') as fp:
            json.dump(uvValDict, fp, indent=4)

        # save albedo scale optimization results
        albScaleDict = {'r': self.albScale[0].item(),
                        'g': self.albScale[1].item(),
                        'b': self.albScale[2].item()}
        with open(self.albScaleSavePath, 'w') as fp:
            json.dump(albScaleDict, fp, indent=4)

        # save the record of error
        np.savetxt(self.errorSavePath, np.array(err_list), fmt='%f')


if __name__ == '__main__':
    # print('')
    parser = argparse.ArgumentParser(description='Material Optimization Script for PhotoScene')
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', required=True, help='first or second round optimization')
    args = parser.parse_args()

    assert (args.mode in ['first', 'second'])

    cfg = load_cfg(args.config)
    selectedGraphDict = readGraphDict(cfg.selectedGraphFile)
    matViewDict = readViewDict(cfg.selectedViewFile)
    refH, refW = cfg.imgHeight, cfg.imgWidth
    for mat, graphName in selectedGraphDict.items():

        print('\n---> Material Optimization for part [%s] with graph [%s]' % (mat, graphName))

        vId = matViewDict[mat]
        isHdr = False
        inputMode = 'photo' if args.mode == 'first' else 'geo'
        resumeMode = None if args.mode == 'first' else 'first'
        fixUv = False if args.mode == 'first' else True
        fixGraph = False if args.mode == 'first' else True

        inputData = loadOptInput(cfg, mat, vId, refH, refW, mode=inputMode, device=cfg.device)

        paramSaveDir = Path(cfg.graphParamSaveDirByName % (mat))
        paramSaveDir.mkdir(parents=True, exist_ok=True)

        matSaveDir = cfg.matPhotoSceneFinalDirByName if args.mode == 'second' else cfg.matPhotoSceneInitDirByName
        matSaveDir = Path(matSaveDir % mat)
        matSaveDir.mkdir(parents=True, exist_ok=True)

        fov = float(getSensorInfoDict(cfg.xmlFile)['fov'])
        renderObj = MicrofacetUV(inputData,
                                    imHeight=refH, imWidth=refW, fov=fov, useVgg=True, useStyle=True,
                                    onlyMean=False, isHdr=isHdr, device=cfg.device)

        if graphName != 'homogeneous':
            translator = MGT(os.path.join(cfg.sbsFileDir, '%s.sbs' % graphName), res=cfg.matRes,
                toolkit_path=cfg.toolkitRoot)
            graph = translator.translate(external_input_folder=Path(cfg.graphDir) / graphName, device='cuda')
            graph.compile()
            optimizer = GraphOptimizer(graph, renderObj, cfg.uvConfigDict, cfg.optObjectiveDict,
                inputData['im'], paramSaveDir, matSaveDir, args.mode, fixUv=fixUv,
                fixGraph=fixGraph, useHomoRough=cfg.useHomoRough, input_seed=cfg.seed)
            sample_num = 20
        else:
            optimizer = HomogeneousOptimizer(renderObj, cfg.optObjectiveDict, inputData['im'], paramSaveDir,
                matSaveDir, args.mode, texRes=2 ** cfg.matRes, input_seed=cfg.seed, device='cuda')
            sample_num = 30

        if resumeMode is None:
            optimizer.init_param_search(sample_num=sample_num)  # search for a better starting point
        else:
            optimizer.load_param(resumeMode)

        optimizer.optimize()

    print('\n---> Material Optimization (%s) is done!\n\n' % args.mode)
