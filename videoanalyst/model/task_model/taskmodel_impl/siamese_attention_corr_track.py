# -*- coding: utf-8 -*

import torch
import torch.nn as nn
from loguru import logger

from videoanalyst.model.common_opr.attention_block import ResidualAttentionLayer
from videoanalyst.model.common_opr.common_block import (conv_bn_relu,
                                                        xcorr_depthwise)
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS)
from videoanalyst.utils import md5sum

torch.set_printoptions(precision=8)


@TRACK_TASKMODELS.register
class SiamAttentionCorrTrack(ModuleBase):
    r"""
    SiamTrack model for tracking

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    head_width: int
        feature width in head structure
    """

    default_hyper_params = dict(
        pretrain_model_path="",
        head_width=256,
        conv_weight_std=0.01,
        neck_conv_bias=[True, True, True, True],
        corr_fea_output=False,
        attention_down_num=2,
        attention_down_k_size=[3, 2],
        attention_down_stride=[2, 2],
        attention_up_num=2,
        attention_up_k_size=[2, 3],
        attention_up_stride=[1, 1],
    )

    def __init__(self, backbone, head, loss=None):
        super(SiamAttentionCorrTrack, self).__init__()
        self.basemodel = backbone
        self.head = head
        self.loss = loss

    def forward(self, *args, phase="train"):
        r"""
        Perform tracking process for different phases (e.g. train / init / track)

        Arguments
        ---------
        target_img: torch.Tensor
            target template image patch
        search_img: torch.Tensor
            search region image patch

        Returns
        -------
        fcos_score_final: torch.Tensor
            predicted score for bboxes, shape=(B, HW, 1)
        fcos_bbox_final: torch.Tensor
            predicted bbox in the crop, shape=(B, HW, 4)
        fcos_cls_prob_final: torch.Tensor
            classification score, shape=(B, HW, 1)
        fcos_ctr_prob_final: torch.Tensor
            center-ness score, shape=(B, HW, 1)
        """
        if phase == 'train':
            # resolve training data
            training_data = args[0]
            target_img = training_data["im_z"]
            search_img = training_data["im_x"]
            # backbone feature
            f_z = self.basemodel(target_img)
            f_x = self.basemodel(search_img)
            # feature adjustment
            c_z_k = self.c_z_k(f_z)
            r_z_k = self.r_z_k(f_z)
            c_x = self.c_x(f_x)
            r_x = self.r_x(f_x)
            # feature matching
            r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)
            # correlation reg residual attention
            r_out = self.reg_attention(r_out)
            # correlation cls residual attention
            c_out = self.cls_attention(c_out)
            # head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                c_out, r_out)
            predict_data = dict(
                cls_pred=fcos_cls_score_final,
                ctr_pred=fcos_ctr_score_final,
                box_pred=fcos_bbox_final,
            )
            if self._hyper_params["corr_fea_output"]:
                predict_data["corr_fea"] = corr_fea
            return predict_data
        elif phase == 'feature':
            target_img, = args
            # backbone feature
            f_z = self.basemodel(target_img)
            # template as kernel
            c_z_k = self.c_z_k(f_z)
            r_z_k = self.r_z_k(f_z)
            # output
            out_list = [c_z_k, r_z_k]
        elif phase == 'track':
            if len(args) == 3:
                search_img, c_z_k, r_z_k = args
                # backbone feature
                f_x = self.basemodel(search_img)
                # feature adjustment
                c_x = self.c_x(f_x)
                r_x = self.r_x(f_x)
            elif len(args) == 4:
                # c_x, r_x already computed
                c_z_k, r_z_k, c_x, r_x = args
            else:
                raise ValueError("Illegal args length: %d" % len(args))

            # feature matching
            r_out = xcorr_depthwise(r_x, r_z_k)
            c_out = xcorr_depthwise(c_x, c_z_k)
            # correlation reg residual attention
            r_out = self.reg_attention(r_out)
            # correlation cls residual attention
            c_out = self.cls_attention(c_out)
            # head
            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(
                c_out, r_out)
            # apply sigmoid
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
            # register extra output
            extra = dict(c_x=c_x, r_x=r_x, corr_fea=corr_fea)
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final, extra
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        r"""
        Load model parameters
        """
        self._make_convs()
        self._initialize_conv()
        self._make_attention()
        self._initialize_attention()
        super().update_params()

        if self._hyper_params["pretrain_model_path"] != "":
            model_path = self._hyper_params["pretrain_model_path"]
            try:
                state_dict = torch.load(model_path,
                                        map_location=torch.device("gpu"))
            except:
                state_dict = torch.load(model_path,
                                        map_location=torch.device("cpu"))
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            try:
                self.load_state_dict(state_dict, strict=True)
            except:
                self.load_state_dict(state_dict, strict=False)
            logger.info("Pretrained weights loaded from {}".format(model_path))
            logger.info("Check md5sum of Pretrained weights: %s" %
                        md5sum(model_path))

    def _make_convs(self):
        head_width = self._hyper_params['head_width']

        # feature adjustment
        self.r_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.c_z_k = conv_bn_relu(head_width,
                                  head_width,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    def _initialize_conv(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        conv_list = [
            self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
        ]
        for ith in range(len(conv_list)):
            conv = conv_list[ith]
            torch.nn.init.normal_(conv.weight,
                                  std=conv_weight_std)  # conv_weight_std=0.01

    def _initialize_attention(self, ):
        conv_weight_std = self._hyper_params['conv_weight_std']
        attention_list = [self.cls_attention, self.reg_attention]
        for ith in range(len(attention_list)):
            attention = attention_list[ith]
            attention.initialize_normal(conv_weight_std)

    def _make_attention(self):
        head_width = self._hyper_params['head_width']
        down_num = self._hyper_params['attention_down_num']
        down_k_size = self._hyper_params['attention_down_k_size']
        down_stride = self._hyper_params['attention_down_stride']
        up_num = self._hyper_params['attention_up_num']
        up_k_size = self._hyper_params['attention_up_k_size']
        up_stride = self._hyper_params['attention_up_stride']
        logger.info("attention layer num: {}, kernel size: {}".format(down_num, down_k_size))
        self.cls_attention = ResidualAttentionLayer(head_width,
                                                    down_k_size=down_k_size,
                                                    down_stride=down_stride,
                                                    up_k_size=up_k_size,
                                                    up_stride=up_stride,
                                                    down_num=down_num,
                                                    up_num=up_num)
        self.reg_attention = ResidualAttentionLayer(head_width,
                                                    down_k_size=down_k_size,
                                                    down_stride=down_stride,
                                                    up_k_size=up_k_size,
                                                    up_stride=up_stride,
                                                    down_num=down_num,
                                                    up_num=up_num)

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
