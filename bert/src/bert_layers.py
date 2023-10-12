# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022, Tri Dao.
# Copyright (c) 2023, MosaicML.
# Copyright (c) 2023, Dan Fu and Simran Arora.

import copy
import logging
import math
import os
import sys
import warnings
from typing import List, Optional, Tuple, Union
from functools import partial

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import bert_padding as bert_padding_module
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (MaskedLMOutput,
                                           SequenceClassifierOutput)
from transformers.models.bert.modeling_bert import BertPreTrainedModel

import pdb

try:
    import flash_attn_triton as flash_attn_triton
    flash_attn_qkvpacked_func = flash_attn_triton.flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None

from src.mm.blockdiag_linear import BlockdiagLinear
from src.mm.monarch_mixer_sequence_mixer import MonarchMixerSequenceMixing

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class BertEmbeddings(nn.Module):
    """Construct the embeddings for words, ignoring position.

    There are no positional embeddings since we use ALiBi and token_type
    embeddings.

    This module is modeled after the Hugging Face BERT's
    :class:`~transformers.model.bert.modeling_bert.BertEmbeddings`, but is
    modified as part of Mosaic BERT's ALiBi implementation. The key change is
    that position embeddings are removed. Position information instead comes
    from attention biases that scale linearly with the position distance
    between query and key tokens.

    This module ignores the `position_ids` input to the `forward` method.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)
        # ALiBi doesn't use position embeddings
        if config.use_positional_encodings:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            #self.position_embeddings = nn.Embedding(512, 960)

            """ concat_positional_embeddings = True
            if concat_positional_embeddings:

                positional_embeddings = self.position_embeddings
                concatenated_output = nn.Embedding(num_embeddings=512, embedding_dim=768, device="cuda:0")
                
                # Copy the weights from the original embedding to the expanded one
                concatenated_output.weight.data[:128] = positional_embeddings.weight.data
                concatenated_output.weight.data[128:256] = positional_embeddings.weight.data
                concatenated_output.weight.data[256:384] = positional_embeddings.weight.data
                concatenated_output.weight.data[384:512] = positional_embeddings.weight.data

                self.position_embeddings = concatenated_output """


        self.use_positional_encodings = config.use_positional_encodings
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.use_positional_encodings:
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer('token_type_ids',
                             torch.zeros(config.max_position_embeddings,
                                         dtype=torch.long),
                             persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        return_position_encodings: bool = False,
    ) -> torch.Tensor:
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Must specify either input_ids or input_embeds!')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            assert inputs_embeds is not None  # just for type checking
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            if self.use_positional_encodings:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor
        # where it is all zeros, which usually occurs when it's auto-generated;
        # registered buffer helps users when tracing the model without passing
        # token_type_ids, solves issue #5664
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                assert isinstance(self.token_type_ids, torch.LongTensor)
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded  # type: ignore
            else:
                token_type_ids = torch.zeros(input_shape,  # type: ignore
                                             dtype=torch.long,
                                             device=self.word_embeddings.device) # type: ignore  # yapf: disable
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        #pdb.set_trace()

        embeddings = inputs_embeds + token_type_embeddings
        if self.use_positional_encodings:
            expanded_embeddings = False
            if not expanded_embeddings:
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            else:
                position_embeddings = self.position_embeddings(position_ids)
                position_embeddings = torch.cat([position_embeddings, position_embeddings, position_embeddings, position_embeddings], axis=1)
                embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        if return_position_encodings:
            return embeddings, position_embeddings
        else:
            return embeddings


class BertUnpadSelfAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Triton is installed, this module uses Flash Attention to greatly improve throughput.
    The Flash Attention implementation used is an adaptation from Mosaic, which supports arbitrary attention biases (
    used to implement ALiBi), but does not support attention dropout. If either Triton is not installed
    or `config.attention_probs_dropout_prob > 0`, the implementation will default to a
    math-equivalent pytorch version, which is much slower.

    See `forward` method for additional detail.
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, 'embedding_size'):
            raise ValueError(
                f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention '
                f'heads ({config.num_attention_heads})')

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(self.all_head_size, 3 * config.hidden_size)

        # Warn if defaulting to pytorch because of import issues
        if flash_attn_qkvpacked_func is None:
            warnings.warn(
                'Unable to import Triton; defaulting attention implementation to pytorch (this will reduce throughput when using this model).'
            )

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_in_batch: int, indices: torch.Tensor,
                attn_mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Perform self-attention.

        If dropout is zero, then we can use the Triton kernel, so we do that. However, if not, we send through a standard PyTorch
        implementation of self-attention.

        The arguments are unpadded, and our implementations of attention require padded arguments,
        so we first call `pad_input`. Once we compute attention, we re-unpad our outputs for the other layers.
        The pad/unpad operations add overhead, but not sending pad tokens through ffs saves compute.
        It is possible to write an unpadded implementation of attention (in Triton and PyTorch), which we will eventually do.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen_in_batch: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen_in_batch)
            bias: (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)

        Returns:
            attention: (total_nnz, dim)
        """
        
        qkv = self.Wqkv(hidden_states)
        qkv = bert_padding_module.pad_input(
            qkv, indices, cu_seqlens.shape[0] - 1,
            max_seqlen_in_batch)  # batch, max_seqlen_in_batch, thd
        qkv = rearrange(qkv,
                        'b s (t h d) -> b s t h d',
                        t=3,
                        h=self.num_attention_heads)
        if self.p_dropout or flash_attn_qkvpacked_func is None:
            # if we have nonzero attention dropout (e.g. during fine-tuning) or no Triton, compute attention in PyTorch
            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d
            k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)  # b h d s
            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d
            attention_scores = torch.matmul(q, k) / math.sqrt(
                self.attention_head_size)
            attention_scores = attention_scores + bias
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            attention = torch.matmul(attention_probs, v).permute(0, 2, 1,
                                                                 3)  # b s h d
        else:
            # Triton implementation only supports 0 attention dropout
            convert_dtype = qkv.dtype not in [torch.float16, torch.bfloat16]
            if convert_dtype:
                # Triton implementation only supports fp16 and bf16
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.float16)
                bias_dtype = bias.dtype
                bias = bias.to(torch.float16)
                attention = flash_attn_qkvpacked_func(qkv, bias)
                attention = attention.to(orig_dtype)
                bias = bias.to(bias_dtype)
            else:
                attention = flash_attn_qkvpacked_func(qkv, bias)

        # attn_mask is 1 for attend and 0 for don't
        attention = bert_padding_module.unpad_input_only(
            attention,
            torch.squeeze(attn_mask) == 1)
        return rearrange(attention, 'nnz h d -> nnz (h d)')


class BertSelfOutput(nn.Module):
    """Computes the output of the attention layer."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                    eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertUnpadAttention(nn.Module):
    """Chains attention, Dropout, and LayerNorm for BERT."""

    def __init__(self, config):
        super().__init__()
        self.self = BertUnpadSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        input_tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for scaled self-attention without padding.

        Arguments:
            input_tensor: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_s: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        """
        self_output = self.self(input_tensor, cu_seqlens, max_s, indices,
                                attn_mask, bias)
        if subset_idx is not None:
            return self.output(
                bert_padding_module.index_first_axis(self_output, subset_idx),
                bert_padding_module.index_first_axis(input_tensor, subset_idx))
        else:
            return self.output(self_output, input_tensor)


class BertMLP(nn.Module):
    """Applies the FFN at the end of each BERT layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.use_monarch_mlp:
            linear_cls = partial(BlockdiagLinear, nblocks=self.config.monarch_mlp_nblocks)
        else:
            linear_cls = nn.Linear

        self.gated_layers = linear_cls(config.hidden_size,
                                        config.intermediate_size,
                                        bias=False)
        self.act = nn.GELU(approximate='none')
        self.wo = linear_cls(config.intermediate_size, config.hidden_size)

        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        
        residual_connection = hidden_states
        hidden_states = self.gated_layers(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states


class BertGatedLinearUnitMLP(nn.Module):
    """Applies the FFN at the end of each BERT layer with a Gated Linear Unit"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.is_padded = config.monarch_mixer_sequence_mixing

        if self.config.use_monarch_mlp:
            linear_cls = partial(BlockdiagLinear, nblocks=self.config.monarch_mlp_nblocks)
        else:
            linear_cls = nn.Linear
        self.gated_layers = linear_cls(
            config.hidden_size,
            config.intermediate_size * 2,
            bias=False
        )
        self.act = nn.GELU(approximate='none')
        self.wo = linear_cls(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """

        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)

        if self.is_padded:
            gated = hidden_states[:, :, :self.config.intermediate_size]
            non_gated = hidden_states[:, :, self.config.intermediate_size:]
        else:
            gated = hidden_states[:, :self.config.intermediate_size]
            non_gated = hidden_states[:, self.config.intermediate_size:]

        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)

        return hidden_states


class BertLayer(nn.Module):
    """BERT layer, which includes Sequence Mixing (e.g. Attention or Hyena) and State Mixing (e.g. MLP)."""

    def __init__(self, config):
        super(BertLayer, self).__init__()

        self.monarch_mixer_sequence_mixing = config.monarch_mixer_sequence_mixing
        
        print(f"Using Monarch Mixer for Sequence Mixing: {config.monarch_mixer_sequence_mixing}")
        if config.monarch_mixer_sequence_mixing:
            if config.use_flash_mm:
                from src.mm.flash_mm import FlashMMSequenceMixing
                mm_cls = FlashMMSequenceMixing
            else:
                mm_cls = MonarchMixerSequenceMixing
            self.attention = mm_cls(
                config.hidden_size,
                l_max=config.long_conv_l_max,
                hyena_kernel_lr=config.long_conv_kernel_learning_rate,
                bidirectional=config.bidirectional,

                hyena_lr_pos_emb=config.hyena_lr_pos_emb,
                hyena_w=config.hyena_w,
                hyena_w_mod=config.hyena_w_mod,
                hyena_wd=config.hyena_wd,
                hyena_emb_dim=config.hyena_emb_dim,
                hyena_filter_dropout=config.hyena_filter_dropout,
                hyena_filter_order=config.hyena_filter_order,
                residual_long_conv=config.residual_long_conv,
                hyena_training_additions=config.hyena_training_additions,
            )

        else:
            self.attention = BertUnpadAttention(config)
        if config.use_glu_mlp:
            self.mlp = BertGatedLinearUnitMLP(config)
        else:
            self.mlp = BertMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seqlen: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            seqlen: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        """

        if self.monarch_mixer_sequence_mixing:
            attention_output = self.attention(hidden_states)
            if type(attention_output) == tuple:
                attention_output, _ = attention_output
        else:
            attention_output = self.attention(hidden_states, cu_seqlens, seqlen,
                                          subset_idx, indices, attn_mask, bias)

        layer_output = self.mlp(attention_output)

        return layer_output


class BertEncoder(nn.Module):
    """A stack of BERT layers providing the backbone of BERT.

    Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
    at padded tokens, and pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self.monarch_mixer_sequence_mixing = config.monarch_mixer_sequence_mixing
        self.num_attention_heads = config.num_attention_heads

        if not self.monarch_mixer_sequence_mixing:
            # The alibi mask will be dynamically expanded if it is too small for
            # the input the model receives. But it generally helps to initialize it
            # to a reasonably large size to help pre-allocate CUDA memory.
            # The default `alibi_starting_size` is 512.
            self._current_alibi_size = int(config.alibi_starting_size)
            self.alibi = torch.zeros(
                (1, self.num_attention_heads, self._current_alibi_size,
                self._current_alibi_size))
            self.rebuild_alibi_tensor(size=config.alibi_starting_size)

    def rebuild_alibi_tensor(self,
                             size: int,
                             device: Optional[Union[torch.device, str]] = None):
        # Alibi
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        n_heads = self.num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> List[float]:

            def get_slopes_power_of_2(n_heads: int) -> List[float]:
                start = (2**(-2**-(math.log2(n_heads) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n_heads)]

            # In the paper, they only train models that have 2^a heads for some a. This function
            # has some good properties that only occur when the input is a power of 2. To
            # maintain that even when the number of heads is not a power of 2, we use a
            # workaround.
            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][:n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(
            n_heads, -1, -1)
        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])

        self._current_alibi_size = size
        self.alibi = alibi

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: Optional[bool] = True,
        subset_mask: Optional[torch.Tensor] = None,
        position_encodings: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask_bool = attention_mask.bool()
        batch, seqlen = hidden_states.shape[:2]

        # Unpad inputs and mask. It will remove tokens that are padded.
        # Assume ntokens is total number of tokens (padded and non-padded)
        # and ntokens_unpad is total number of non-padded tokens.
        # Then unpadding performs the following compression of the inputs:
        # hidden_states[ntokens,hidden] -> hidden_states[ntokens_unpad,hidden]
        if not self.monarch_mixer_sequence_mixing:
            hidden_states, indices, cu_seqlens, _ = bert_padding_module.unpad_input(
                hidden_states, attention_mask_bool)
        else:
            cu_seqlens = None
            indices = None

        # Add alibi matrix to extended_attention_mask
        if not self.monarch_mixer_sequence_mixing:
            if self._current_alibi_size < seqlen:
                # Rebuild the alibi tensor when needed
                warnings.warn(
                    f'Increasing alibi size from {self._current_alibi_size} to {seqlen}'
                )
                self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
            elif self.alibi.device != hidden_states.device:
                # Device catch-up
                self.alibi = self.alibi.to(hidden_states.device)
            alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
            attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
            alibi_attn_mask = attn_bias + alibi_bias
        else:
            alibi_attn_mask = None

        all_encoder_layers = []
        if self.monarch_mixer_sequence_mixing:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states,
                    cu_seqlens,
                    seqlen,
                    None,
                    indices,
                    attn_mask=attention_mask,
                    bias=alibi_attn_mask
                )
                if position_encodings is not None:
                    hidden_states = hidden_states + position_encodings
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            if subset_mask is not None:
                hidden_states = hidden_states[subset_mask]
        else:
            if subset_mask is None:
                for layer_module in self.layer:
                    hidden_states = layer_module(hidden_states,
                        cu_seqlens,
                        seqlen,
                        None,
                        indices,
                        attn_mask=attention_mask,
                        bias=alibi_attn_mask
                    )
                    if output_all_encoded_layers:
                        all_encoder_layers.append(hidden_states)

                # Pad inputs and mask. It will insert back zero-padded tokens.
                # Assume ntokens is total number of tokens (padded and non-padded)
                # and ntokens_unpad is total number of non-padded tokens.
                # Then padding performs the following de-compression:
                #     hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
                hidden_states = bert_padding_module.pad_input(
                    hidden_states, indices, batch, seqlen
                )
            else:
                for i in range(len(self.layer) - 1):
                    layer_module = self.layer[i]
                    hidden_states = layer_module(hidden_states,
                                                    cu_seqlens,
                                                    seqlen,
                                                    None,
                                                    indices,
                                                    attn_mask=attention_mask,
                                                    bias=alibi_attn_mask)
                    if output_all_encoded_layers:
                        all_encoder_layers.append(hidden_states)
                subset_idx = torch.nonzero(subset_mask[attention_mask_bool],
                                            as_tuple=False).flatten()
                    
                hidden_states = self.layer[-1](hidden_states,
                                                cu_seqlens,
                                                seqlen,
                                                subset_idx=subset_idx,
                                                indices=indices,
                                                attn_mask=attention_mask,
                                                bias=alibi_attn_mask)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    

class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.pool_all = config.pool_all

    def forward(self,
                hidden_states: torch.Tensor,
                pool: Optional[bool] = True,
                mask= None) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        if not self.pool_all:
            first_token_tensor = hidden_states[:, 0] if pool else hidden_states
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)
        else:
            # mean pool everything that isn't masked out
            denom = torch.sum(mask, dim=1, keepdim=True)
            mean_tensor = torch.sum((hidden_states) * mask.unsqueeze(-1), dim = 1) / denom
            pooled_output = self.dense(mean_tensor)
            pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertModel(BertPreTrainedModel):
    """Overall BERT model.

    Args:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controlled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.post_init()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_all_encoded_layers: Optional[bool] = False,
        masked_tokens_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(
            input_ids, 
            token_type_ids,
            position_ids
        )
        position_encodings = None

        subset_mask = []
        first_col_mask = []

        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            subset_mask=subset_mask,
            position_encodings=position_encodings)
        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[-1]
            pooled_output = self.pooler(
                sequence_output, mask = attention_mask) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask.bool()
            subset_idx = subset_mask[attention_mask_bool]  # type: ignore
            sequence_output = encoder_outputs[-1][
                masked_tokens_mask[attention_mask_bool][subset_idx]]
            if self.pooler is not None:
                pool_input = encoder_outputs[-1][
                    first_col_mask[attention_mask_bool][subset_idx]]
                pooled_output = self.pooler(pool_input, pool=False, mask = attention_mask)
            else:
                pooled_output = None

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output

        if self.pooler is not None:
            return encoder_outputs, pooled_output

        return encoder_outputs, None


###################
# Bert Heads
###################
class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0))
        self.decoder.weight = bert_model_embedding_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


#######################
# Construct Bert model
#######################
class BertForMaskedLM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            warnings.warn(
                'If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for '
                'bi-directional self-attention.')

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config,
                                   self.bert.embeddings.word_embeddings.weight)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_composer(cls,
                      pretrained_checkpoint,
                      state_dict=None,
                      cache_dir=None,
                      from_tf=False,
                      config=None,
                      *inputs,
                      **kwargs):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError(
                'TensorFlow is not supported.')

        state_dict = torch.load(pretrained_checkpoint)
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix
        #print("state_dict found!")
        #print(state_dict.keys())

        consume_prefix_in_state_dict_if_present(state_dict, prefix='model.')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                              strict=False)

        if len(missing_keys) > 0:
            logger.warning(
                f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}"
            )
        if len(unexpected_keys) > 0:
            logger.warning(
                f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}"
            )

        return model

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        # labels should be a `torch.LongTensor` of shape
        # `(batch_size, sequence_length)`. These are used for computing the
        #  masked language modeling loss.
        #
        # Indices should be in `[-100, 0, ..., config.vocab_size]` (see
        # `input_ids` docstring) Tokens with indices set to `-100` are ignored
        # (masked), the loss is only computed for the tokens with labels in `[0,
        # ..., config.vocab_size]`
        #
        # Prediction scores are only computed for masked tokens and the (bs,
        # seqlen) dimensions are flattened
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Must specify either input_ids or input_embeds!')

        if labels is None: 
            masked_tokens_mask = None
        else:
            masked_tokens_mask = labels > 0

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            masked_tokens_mask=masked_tokens_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        loss = None
        if labels is not None:
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()

            masked_token_idx = torch.nonzero(labels.flatten() > 0,
                                                as_tuple=False).flatten()
            loss = loss_fct(prediction_scores,
                                labels.flatten()[masked_token_idx])
            assert input_ids is not None, 'Coding error; please open an issue'
            batch, seqlen = input_ids.shape[:2]
            prediction_scores = rearrange(
                bert_padding_module.index_put_first_axis(
                prediction_scores, masked_token_idx, batch * seqlen),
                '(b s) d -> b s d',
                b=batch)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor,
                                      attention_mask: torch.Tensor,
                                      **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError('The PAD token should be defined for generation')

        attention_mask = torch.cat([
            attention_mask,
            attention_mask.new_zeros((attention_mask.shape[0], 1))
        ], dim=-1)
        dummy_token = torch.full((effective_batch_size, 1),
                                 self.config.pad_token_id,
                                 dtype=torch.long,
                                 device=input_ids.device)
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class BertForSequenceClassification(BertPreTrainedModel):
    """Bert Model transformer with a sequence classification/regression head.

    This head is just a linear layer on top of the pooled output. Used for,
    e.g., GLUE tasks.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_composer(cls,
                      pretrained_checkpoint,
                      state_dict=None,
                      cache_dir=None,
                      from_tf=False,
                      config=None,
                      *inputs,
                      **kwargs):
        """Load from pre-trained."""
        model = cls(config, *inputs, **kwargs)
        if from_tf:
            raise ValueError(
                'TensorFlow is not supported.')
        
        ############################################################

        state_dict = torch.load(pretrained_checkpoint)
        state_dict = state_dict['state']['model']
        # If the state_dict was saved after wrapping with `composer.HuggingFaceModel`, it takes on the `model` prefix

        #print("Keys Found")
        #for key in state_dict.keys():
        #    #if "pos_emb.t" in key:
        #    if "delta" in key:
        #        print(key)
        #        print(state_dict[key].shape)
        
        #pdb.set_trace()

        ########################################

        expand_positional_embeddings = config.expand_positional_embeddings
        if expand_positional_embeddings:

            print("Expanding positional embeddings")

            original_embedding = state_dict['model.bert.embeddings.position_embeddings.weight']
            #state_dict['model.bert.embeddings.position_embeddings.weight'] = torch.cat([original_embedding, original_embedding, original_embedding, original_embedding], axis=0)
            #pdb.set_trace()
            import numpy as np
            position_embeddings_randomized_1 = torch.from_numpy(np.random.uniform(-1, 1, size=(original_embedding.shape[0], original_embedding.shape[1]))).to(torch.device("cuda:0"))
            position_embeddings_randomized_2 = torch.from_numpy(np.random.uniform(-1, 1, size=(original_embedding.shape[0], original_embedding.shape[1]))).to(torch.device("cuda:0"))
            position_embeddings_randomized_3 = torch.from_numpy(np.random.uniform(-1, 1, size=(original_embedding.shape[0], original_embedding.shape[1]))).to(torch.device("cuda:0"))
            state_dict['model.bert.embeddings.position_embeddings.weight'] = torch.cat([original_embedding, position_embeddings_randomized_1, position_embeddings_randomized_2, position_embeddings_randomized_3], axis=0)

            assert state_dict['model.bert.embeddings.position_embeddings.weight'].shape[0] == 512
            assert state_dict['model.bert.embeddings.position_embeddings.weight'].shape[1] in [768, 960, 1536, 1792]

            print("position_ids")
            print(state_dict['model.bert.embeddings.position_ids'])
            del state_dict['model.bert.embeddings.position_ids']

            for i in range(0, 12):

                def expand_parameter(current_param):
                    #return torch.cat([current_param, current_param, current_param, current_param], axis=1)
                    expanded_parameter = nn.Parameter(torch.zeros(current_param.shape[0], 4 * current_param.shape[1], current_param.shape[2]))
                    expanded_parameter.data[:, :current_param.shape[1], :] = current_param
                    expanded_parameter.data[:, current_param.shape[1]: 2 * current_param.shape[1], :] = current_param
                    expanded_parameter.data[:, 2 * current_param.shape[1]: 3 * current_param.shape[1], :] = current_param
                    expanded_parameter.data[:, 3 * current_param.shape[1]: 4 * current_param.shape[1], :] = current_param
                    return expanded_parameter

                state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn.pos_emb.z'] = expand_parameter(state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn.pos_emb.z'])
                assert state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn.pos_emb.z'].shape[1] == 512
                
                state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn2.pos_emb.z'] = expand_parameter(state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn2.pos_emb.z'])
                assert state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn2.pos_emb.z'].shape[1] == 512

                #del state_dict["model.bert.encoder.layer." + str(i) + ".attention.filter_fn.pos_emb.t"]
                #del state_dict["model.bert.encoder.layer." + str(i) + ".attention.filter_fn2.pos_emb.t"]

                #pdb.set_trace()

                def expand_parameter_v2(current_param):
                    #return torch.cat([current_param, current_param, current_param, current_param], axis=1)
                    expanded_parameter = nn.Parameter(torch.ones(current_param.shape[0], 4 * current_param.shape[1], current_param.shape[2]))
                    expanded_parameter.data[:, :current_param.shape[1], :] = current_param
                    return expanded_parameter
                
                state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn.pos_emb.t'] = expand_parameter_v2(state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn.pos_emb.t'])
                assert state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn.pos_emb.t'].shape[1] == 512
                
                state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn2.pos_emb.t'] = expand_parameter_v2(state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn2.pos_emb.t'])
                assert state_dict['model.bert.encoder.layer.' + str(i) + '.attention.filter_fn2.pos_emb.t'].shape[1] == 512

        ###################################################

        consume_prefix_in_state_dict_if_present(state_dict, prefix='model.')

        print("position_ids in model")
        print(model.bert.embeddings.position_ids)
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, 
                                                              strict=False)
        
        print("Initiliazed and loaded BERT model")
        print(model)
        print("model.bert.encoder.layer[10].attention.filter_fn2.pos_emb.t")
        print(model.bert.encoder.layer[10].attention.filter_fn2.pos_emb.t.shape)
        print(model.bert.encoder.layer[10].attention.filter_fn2.pos_emb.t)

        ########################################

        """expand_positional_embeddings = True
        if expand_positional_embeddings:
            
            original_embedding = model.bert.embeddings.position_embeddings
            new_num_embeddings = 4 * original_embedding.num_embeddings
            expanded_embedding = nn.Embedding(num_embeddings=new_num_embeddings, embedding_dim=original_embedding.embedding_dim)

            expanded_embedding.weight.data[0:128] = original_embedding.weight.data[0:128]
            expanded_embedding.weight.data[128:256] = original_embedding.weight.data[0:128]
            expanded_embedding.weight.data[256:384] = original_embedding.weight.data[0:128]
            expanded_embedding.weight.data[384:512] = original_embedding.weight.data[0:128]

            model.bert.embeddings.position_embeddings = expanded_embedding
            assert expanded_embedding.weight.shape[0] == 512
            assert expanded_embedding.weight.shape[1] in [768, 960, 1536, 1792]

            original_position_ids = model.bert.embeddings.position_ids
            model.bert.embeddings.position_ids = torch.cat([original_position_ids, original_position_ids, original_position_ids, original_position_ids], axis=1)
            assert model.bert.embeddings.position_ids.shape[0] == 1
            assert model.bert.embeddings.position_ids.shape[1] == 512

            for i in range(0, 12):

                def expand_parameter(current_param):
                    expanded_parameter = nn.Parameter(torch.zeros(current_param.shape[0], 4 * current_param.shape[1], current_param.shape[2]))
                    expanded_parameter.data[:, :current_param.shape[1], :] = current_param.data
                    expanded_parameter.data[:, current_param.shape[1]: 2 * current_param.shape[1], :] = current_param.data
                    expanded_parameter.data[:, 2 * current_param.shape[1]: 3 * current_param.shape[1], :] = current_param.data
                    expanded_parameter.data[:, 3 * current_param.shape[1]: 4 * current_param.shape[1], :] = current_param.data
                    return expanded_parameter

                model.bert.encoder.layer[i].attention.filter_fn.pos_emb.z = expand_parameter(model.bert.encoder.layer[i].attention.filter_fn.pos_emb.z)
                assert model.bert.encoder.layer[i].attention.filter_fn.pos_emb.z.shape[1] == 512
                
                model.bert.encoder.layer[i].attention.filter_fn2.pos_emb.z = expand_parameter(model.bert.encoder.layer[i].attention.filter_fn2.pos_emb.z)
                assert model.bert.encoder.layer[i].attention.filter_fn2.pos_emb.z.shape[1] == 512

                #model.bert.encoder.layer[i].attention.filter_fn.pos_emb.t = expand_parameter(model.bert.encoder.layer[i].attention.filter_fn.pos_emb.t)
                #assert model.bert.encoder.layer[i].attention.filter_fn.pos_emb.t.shape[1] == 512
                
                #model.bert.encoder.layer[i].attention.filter_fn2.pos_emb.t = expand_parameter(model.bert.encoder.layer[i].attention.filter_fn2.pos_emb.t)
                #assert model.bert.encoder.layer[i].attention.filter_fn2.pos_emb.t.shape[1] == 512

                ###################################################

            print("Manipulated BERT model")
            print(model)
            #assert False"""

        ################################################


        if len(missing_keys) > 0:
            logger.warning(
                f"\nFound these missing keys in the checkpoint: {', '.join(missing_keys)}\n"
            )
        if len(unexpected_keys) > 0:
            logger.warning(
                f"\nFound these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}\n"
            )

        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        # Labels for computing the sequence classification/regression loss.
        # Indices should be in `[0, ..., config.num_labels - 1]`.
        # If `config.num_labels == 1` a regression loss is computed
        # (mean-square loss). If `config.num_labels > 1` a classification loss
        # is computed (cross-entropy).

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        #pdb.set_trace()

        expanding_contexts = self.config.expand_positional_embeddings
        if expanding_contexts:
            assert self.bert.embeddings.position_embeddings.weight.shape[0] == 512
            assert self.bert.embeddings.position_embeddings.weight.shape[1] in [768, 960, 1536, 1792]
            assert self.bert.embeddings.position_ids.shape[0] == 1
            assert self.bert.embeddings.position_ids.shape[1] == 512

            assert self.bert.encoder.layer[0].attention.filter_fn.pos_emb.z.shape[1] == 512
            assert self.bert.encoder.layer[0].attention.filter_fn2.pos_emb.z.shape[1] == 512
            assert self.bert.encoder.layer[11].attention.filter_fn.pos_emb.z.shape[1] == 512
            assert self.bert.encoder.layer[11].attention.filter_fn2.pos_emb.z.shape[1] == 512
            #assert False
        
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # Compute loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or
                                              labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                if self.num_labels == 2 and False: #and self.config.name != "mosaic_bert":
                    loss_fct = nn.BCELoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels),
                                    labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = nn.BCEWithLogitsLoss()
                #loss_fct = nn.CrossEntropyLoss()
                pdb.set_trace()
                loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

