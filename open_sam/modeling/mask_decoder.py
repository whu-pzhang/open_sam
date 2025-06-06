from typing import List, Tuple, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from mmcv.cnn.bricks import build_activation_layer

from open_sam.registry import MODELS
from .common import LayerNorm2d


@MODELS.register_module()
class MaskDecoder(nn.Module):

    def __init__(self,
                 *,
                 transformer_dim: int,
                 transformer: dict,
                 num_multimask_outputs: int = 3,
                 act_cfg: dict = dict(type='GELU'),
                 iou_head_depth: int = 3,
                 iou_head_hidden_dim: int = 256) -> None:
        """Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Borrowed from https://github.com/facebookresearch/segment-anything

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = MODELS.build(transformer)

        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        activation = build_activation_layer(act_cfg)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim,
                               transformer_dim // 4,
                               kernel_size=2,
                               stride=2), LayerNorm2d(transformer_dim // 4),
            activation,
            nn.ConvTranspose2d(transformer_dim // 4,
                               transformer_dim // 8,
                               kernel_size=2,
                               stride=2), activation)
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for i in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim,
                                       self.num_mask_tokens, iou_head_depth)

    def forward(
            self,
            image_embeddings: Tensor,
            image_pe: Tensor,
            sparse_prompt_embeddings: Tensor,
            dense_prompt_embeddings: Tensor,
            multimask_output: bool,
            output_attentions: Optional[bool] = None,
            attention_similarity: torch.Tensor = None,
            target_embedding: torch.Tensor = None) -> Tuple[Tensor, Tensor]:
        """Predict masks given image and prompt embeddings.

        Borrowed from https://github.com/facebookresearch/segment-anything

        Arguments:
            image_embeddings (Tensor): the embeddings from the image encoder
            image_pe (Tensor): positional encoding with 
                the shape of image_embeddings
            sparse_prompt_embeddings (Tensor): the embeddings of
                the points and boxes
            dense_prompt_embeddings (Tensor): the embeddings of the mask inputs
            multimask_output (bool): Whether to return multiple masks or
                a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all
                attention layers.

        Returns:
          Tensor: batched predicted masks
          Tensor: batched predictions of mask quality
        """
        # batch_size, num_channels, height, width = image_embeddings.shape
        # point_batch_size = sparse_prompt_embeddings.shape[1]

        # # Concatenate output tokens
        # output_tokens = torch.cat(
        #     [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1,
        #                                      1)

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Predicts masks.

        See 'forward' for more details.
        """
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape  # B,256,64,64

        # Run the transformer
        # hs is the transformer output for prompt
        # src is the transformer output for image
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        # B,N,C -> B,C,H,W
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](
                mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        # Dot product of the MLP output for each "output token" and
        # the upscaled mask (each output token represent a mask)
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
