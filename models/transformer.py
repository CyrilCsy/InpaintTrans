import torch
import copy
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List


class Transformer(nn.Module):
    def __init__(self, dim_model=512, num_attn_head=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", normalize_before=False, is_sem_embed=False, is_reshape=False):
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(dim_model, num_attn_head, dim_feedforward, dropout, activation,
                                                normalize_before, is_sem_embed)
        encoder_norm = nn.LayerNorm(dim_model) if normalize_before else None

        decoder_layer = TransformerDecoderLayer(dim_model, num_attn_head, dim_feedforward, dropout, activation,
                                                normalize_before, is_sem_embed)
        decoder_norm = nn.LayerNorm(dim_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.dim_model = dim_model
        self.num_attn_head = num_attn_head
        self.num_encoder_layers = num_encoder_layers
        self.is_reshape = is_reshape

    def _reset_parameters(self):  # init weight with xaiver_uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, src_pos_embed, tgt_pos_embed, sem_embed=None):

        src_mask = _mask_to_bool(src_mask)
        tgt_mask = _mask_to_bool(tgt_mask)

        b, c, h, w = [0, 0, 0, 0]
        if self.is_reshape:
            # flatten x BxCxHxW to BxHWxC
            #         mask BxHxW to BxHW
            b, c, h, w = src.shape
            src = src.flatten(2).permute(0, 2, 1)
            tgt = tgt.flatten(2).permute(0, 2, 1)
            if src_mask is not None:
                src_mask = src_mask.flatten(2).permute(0, 2, 1)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.flatten(2).permute(0, 2, 1)
            src_pos_embed = src_pos_embed.flatten(2).permute(0, 2, 1)
            tgt_pos_embed = tgt_pos_embed.flatten(2).permute(0, 2, 1)
            if sem_embed is not None:
                sem_embed = sem_embed.flatten(2).permute(0, 2, 1)

        memory, _ = self.encoder(src, src_mask, src_pos_embed, sem_embed)
        output = self.decoder(tgt, src, tgt_mask, tgt_pos_embed, src_pos_embed, sem_embed)

        if self.is_reshape:
            output = output.permute(1, 2, 0).reshape(b, c, h, w)

        # attn_map = None
        # for attn_weight in attn_weights:
        #     #  attn_map = (attn_map + attn_weight * (torch.abs(self.ratio)+1e-7))/(attn_map+(torch.abs(self.ratio)+1e-7))
        #     attn_map = torch.abs(self.ratio) * attn_weight + (1 - torch.abs(self.ratio)) * attn_map if attn_map is not None else attn_weight
        # attn_map = torch.unsqueeze(attn_map, dim=1).reshape(bs, 1, h*w, h*w)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, mask, pos, sem=None):
        output = x
        # for layer, mask in zip(self.layers, masks):
        for layer in self.layers:
            output, _ = layer(output, mask, pos, sem)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_attn_head, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, is_sem_embed=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, num_attn_head, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.is_sem_embed = is_sem_embed

    def forward_post(self,
                     x,
                     mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     sem: Optional[Tensor] = None):
        q = k = with_sem_embed(with_pos_embed(x, pos), sem)
        v = x

        """
        Args:
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
                value will be ignored.
        """
        x2, _ = self.self_attn(query=q, key=k, value=v,
                                         key_padding_mask=mask)

        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

    def forward_pre(self, x,
                    mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    sem: Optional[Tensor] = None):
        x2 = self.norm1(x)
        q = k = with_sem_embed(with_pos_embed(x2, pos), sem)
        v = x2

        x2, _ = self.self_attn(query=q, key=k, value=v,
                                         key_padding_mask=mask)

        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x

    def forward(self, x,
                mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                sem: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(x, mask, pos, sem)
        return self.forward_post(x, mask, pos, sem)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                x, memory,
                mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                sem: Optional[Tensor] = None
                ):
        output = x
        for layer in self.layers:
            output, attn_weights = layer(output, memory, mask, pos, memory_pos, sem)
        if self.norm is not None:
            output = self.norm(output)
        return output, attn_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_model, num_attn_head, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, is_sem_embed=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, num_attn_head, dropout=dropout, batch_first=True)
        self.multi_attn = nn.MultiheadAttention(dim_model, num_attn_head, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self,
                     x, memory,
                     mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     memory_pos: Optional[Tensor] = None,
                     sem: Optional[Tensor] = None
                     ):
        q = k = with_sem_embed(with_pos_embed(x, pos), sem)
        v = x
        x2, attn_weights = self.self_attn(query=q, key=k, value=v, key_padding_mask=mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        q = with_pos_embed(x, pos)
        k = with_pos_embed(memory, memory_pos)
        v = memory
        x2, _ = self.multi_attn(query=q, key=k, value=v)
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)
        return x, attn_weights

    def forward_pre(self,
                    x, memory,
                    mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    memory_pos: Optional[Tensor] = None,
                    sem: Optional[Tensor] = None
                    ):
        x2 = self.norm1(x)
        q = k = with_sem_embed(with_pos_embed(x2, pos), sem)
        v = x2
        x2, _ = self.self_attn(query=q, key=k, value=v, key_padding_mask=mask)
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)

        q = with_pos_embed(x2, pos)
        k = with_pos_embed(memory, memory_pos)
        v = memory
        x2, _ = self.multi_attn(query=q, key=k, value=v)
        x = x + self.dropout2(x2)
        x2 = self.norm3(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout3(x2)
        return x

    def forward(self,
                x, memory,
                mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                sem: Optional[Tensor] = None
                ):
        if self.normalize_before:
            return self.forward_pre(x, memory, mask, pos, memory_pos, sem)
        return self.forward_post(x, memory, mask, pos, memory_pos, sem)


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


def with_sem_embed(tensor, sem: Optional[Tensor]):
    return tensor if sem is None else tensor + sem


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _mask_to_bool(mask):
    if mask is not None and mask.dtype != torch.bool:
        mask = ~mask.to(torch.bool)
    return mask


def build_transformer(config):
    return Transformer(dim_model=config.dim_model,
                       num_attn_head=config.nhead,
                       num_encoder_layers=config.enc_layers,
                       num_decoder_layers=config.dec_layers,
                       dim_feedforward=config.dim_feedforward,
                       dropout=config.dropout,
                       activation=config.activation_trans,
                       normalize_before=config.norm_pre,
                       is_sem_embed=config.is_sem_embed,
                       is_reshape=config.is_reshape)


def build_transformer_decoder(config):
    decoder_layer = TransformerDecoderLayer(dim_model=config.dim_model,
                                            num_attn_head=config.nhead,
                                            dim_feedforward=config.dim_feedforward,
                                            dropout=config.dropout,
                                            activation=config.activation_trans,
                                            normalize_before=config.norm_pre,
                                            is_sem_embed=config.is_sem_embed)
    return TransformerDecoder(decoder_layer=decoder_layer,
                              num_layers=config.dec_layers,
                              norm=None)
