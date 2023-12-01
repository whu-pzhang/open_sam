from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from mmcv.cnn.bricks import build_activation_layer

from open_sam.registry import MODELS
from .common import LayerNorm2d

from .mask_decoder import MaskDecoder, MLP


@MODELS.register_module()
class HQSAMMaskDecoderHQ(MaskDecoder):

    def __init__(self, vit_embed_dim=768, **kwargs):
        super().__init__(**kwargs)

        self.hf_token = nn.Embedding(1, self.transformer_dim)
        self.hf_mlp = MLP(self.transformer_dim, self.transformer_dim,
                          self.transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_embed_dim,
                               self.transformer_dim,
                               kernel_size=2,
                               stride=2), LayerNorm2d(self.transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.transformer_dim,
                               self.transformer_dim // 8,
                               kernel_size=2,
                               stride=2))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(self.transformer_dim,
                               self.transformer_dim // 4,
                               kernel_size=2,
                               stride=2),
            LayerNorm2d(self.transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(self.transformer_dim // 4,
                               self.transformer_dim // 8,
                               kernel_size=2,
                               stride=2),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(self.transformer_dim // 8, self.transformer_dim // 4, 3,
                      1, 1), LayerNorm2d(self.transformer_dim // 4), nn.GELU(),
            nn.Conv2d(self.transformer_dim // 4, self.transformer_dim // 8, 3,
                      1, 1))

    def forward(
            self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor, multimask_output: bool,
            hq_token_only: bool, interm_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """
        # early-layer ViT feature, after 1st global attention block in ViT
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)

        hq_features = self.embedding_encoder(
            image_embeddings) + self.compress_vit_feat(vit_features)

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature=hq_features[i_batch].unsqueeze(0))
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks, 0)
        iou_preds = torch.cat(iou_preds, 0)

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),
                                    max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]

        masks_hq = masks[:,
                         slice(self.num_mask_tokens -
                               1, self.num_mask_tokens), :, :]

        if hq_token_only:
            return masks_hq
        else:
            return masks_sam, masks_hq

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([
            self.iou_token.weight, self.mask_tokens.weight,
            self.hf_token.weight
        ],
                                  dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(
            upscaled_embedding_sam) + hq_feature

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](
                    mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (
            hyper_in[:, :4] @ upscaled_embedding_sam.view(b, c, h * w)).view(
                b, -1, h, w)
        masks_ours = (
            hyper_in[:, 4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(
                b, -1, h, w)
        masks = torch.cat([masks_sam, masks_ours], dim=1)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
