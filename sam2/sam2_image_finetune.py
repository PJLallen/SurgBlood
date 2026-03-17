# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL.Image import Image

from sam2.modeling.sam2_base import SAM2Base

from sam2.utils.transforms import SAM2Transforms


class SAM2ImagePredictor(torch.nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          fill_hole_area (int): If fill_hole_area > 0, we fill small holes in up to
            the maximum area of fill_hole_area in low_res_masks.
        """
        super().__init__()
        self.model = sam_model
        self._transforms = SAM2Transforms(
            resolution=self.model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # # Spatial dim for backbone feature maps 1024
        # self._bb_feat_sizes = [
        #     (256, 256),
        #     (128, 128),
        #     (64, 64),
        # ]
        # Spatial dim for backbone feature maps  256
        self._bb_feat_sizes = [
            (64, 64),
            (32, 32),
            (16, 16),
        ]

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model)

    def set_image_batch(
        self,
        img_batch,
        # image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        # assert isinstance(image_list, list)
        self._orig_hw = []
        # for image in image_list:
        #     assert isinstance(
        #         image, np.ndarray
        #     ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
        #     self._orig_hw.append(image.shape[:2])

        # # Transform the image to the form expected by the model
        # img_batch = self._transforms.forward_batch(image_list)
        # img_batch = img_batch.to(self.device)

        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"

        logging.info("Computing image embeddings for the provided images...")

        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]


        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}#最后一层?

        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")



    def forward(self, batched_input, multimask_output, image_size=None):
        if len(batched_input) == 1:##

            outputs = self.forward_test(batched_input, multimask_output, image_size)
        else:

            outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs



    def forward_train(
        self,
        batched_input,
        multimask_output: bool = False,
        image_size=256,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        self.set_image_batch(batched_input)#经过backbone,得到并保存特征图_features

        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=None,boxes=None,masks=None,
        )


        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        low_res_masks, iou_predictions,sam_tokens_out, obscured_score_logits = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )

        # # Upscale the masks to the original image resolution
        # masks = self._transforms.postprocess_masks(
        #     low_res_masks, self._orig_hw[img_idx]
        # )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, [256,256]
        )

        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)

        # print(masks.shape)
        # print(low_res_masks.shape)
        # print(iou_predictions.shape)
        mask=masks
        low_res_mask=low_res_masks
        point_predictions=iou_predictions

        # return masks, iou_predictions, low_res_masks, obscured_score_logits
        return mask, point_predictions, low_res_mask, obscured_score_logits


    @torch.no_grad()
    def set_image(self,
                  img_batch,
                  # image: Union[np.ndarray, Image],
                ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # # Transform the image to the form expected by the model
        # if isinstance(image, np.ndarray):
        #     logging.info("For numpy array image, we assume (HxWxC) format")
        self._orig_hw = [img_batch.shape[:2]]
        # elif isinstance(image, Image):
        #     w, h = image.size
        #     self._orig_hw = [(h, w)]
        # else:
        #     raise NotImplementedError("Image format not supported")

        # input_image = self._transforms(image)
        # input_image = input_image[None, ...].to(self.device)

        # assert (
        #     len(input_image.shape) == 4 and input_image.shape[1] == 3
        # ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        # logging.info("Computing image embeddings for the provided image...")

        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        logging.info("Image embeddings computed.")




    def forward_test(
            self,
            batched_input,
            multimask_output: bool = False,
            image_size=256,
            return_logits: bool = False,
            normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        


        self.set_image(batched_input)  # batch=1,经过backbone,得到并保存特征图_features

        # assert self._is_batch, "This function should only be used when in batched mode"
        # if not self._is_image_set:
        #     raise RuntimeError(
        #         "An image must be set with .set_image_batch(...) before mask prediction."
        #     )

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=None,boxes=None,masks=None,
        )


        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        low_res_masks, iou_predictions,sam_tokens_out, obscured_score_logits = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )

        # # Upscale the masks to the original image resolution
        # masks = self._transforms.postprocess_masks(
        #     low_res_masks, self._orig_hw[img_idx]
        # )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, [512,512]
        )

        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        # if not return_logits:#这里意味着
        #     masks = masks > self.mask_threshold

        # print("finetne")
        # print(masks.shape)
        # print(low_res_masks.shape)
        print("---------------------------------------------------------------")
        print(iou_predictions)
        mask=masks
        low_res_mask=low_res_masks
        point_predictions=iou_predictions

        # return masks, iou_predictions, low_res_masks, obscured_score_logits
        return mask, point_predictions, low_res_mask, obscured_score_logits




    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_predictor(self) -> None:
        """
        Resets the image embeddings and other state variables.
        """
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
