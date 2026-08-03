import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from collections.abc import Sequence
from typing import Literal, TypeAlias

from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
from openpi.models.gemma import Config
from openpi.models.gemma import Embedder
from openpi.models.gemma import KVCache
from openpi.models.gemma import RMSNorm
from openpi.models.gemma import _apply_rope
from openpi.models.gemma import _gated_residual
from openpi.models.gemma import Attention
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding


from mme_vla_suite.models.representation.utils import kernel_init_out_proj
from mme_vla_suite.models.representation.robottt import RoboTTTLayer, create_fast_state
from mme_vla_suite.models.integration.utils import _name
from mme_vla_suite.models.integration.utils import Attention_with_MemoryExpert
from mme_vla_suite.models.integration.utils import get_config, Variant

@at.typecheck
class MemoryRMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype)
        
        # modulation = nn.Dense(x.shape[-1] * 2, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        modulation = nn.Dense(x.shape[-1] * 2, kernel_init=kernel_init_out_proj, dtype=dtype)(cond) # add small randomness instead of pure zeros
        scale, shift = jnp.split(modulation, 2, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype)

class MemoryAttention(nn.Module):
    """
    Cross Attention for Memory Modulation
    Use action sequence to attend memory sequence.
    """
    @nn.compact
    def __call__(self, x, mem_seq, mem_mask):
        # x: [B, T, D], mem_seq: [B, S, D], mem_mask: [B, S]
        B, mem_len, mem_width = mem_seq.shape
        B, x_len, x_width = x.shape
        # Let's hardcode the values for now
        num_heads, num_kv_heads, head_dim, width = (
            4,
            1,
            256,
            1024,
        )  # same dim as the action expert in pi05
        assert mem_width == x_width == width
        q_einsum = lora.Einsum(
            shape=(num_heads, width, head_dim),
            name="q_einsum_mem",
            init_fn=nn.initializers.lecun_normal(
                in_axis=-2, out_axis=-1, batch_axis=(0,)
            ),
        )
        kv_einsum = lora.Einsum(
            shape=(2, num_kv_heads, width, head_dim),
            name="kv_einsum_mem",
            init_fn=nn.initializers.lecun_normal(
                in_axis=-2, out_axis=-1, batch_axis=(0, 1)
            ),
        )
        rms_norm = MemoryRMSNorm(name="mem_rms_norm")
        x = rms_norm(x)
        q = q_einsum("BTD,NDH->BTNH", x)
        
        mem_seq = rms_norm(mem_seq)
        k, v = kv_einsum("BSD,2KDH->2BSKH", mem_seq)
        
        q_positions = einops.repeat(
            jnp.arange(mem_len, x_len + mem_len), "t -> b t", b=B
        )
        k_positions = einops.repeat(jnp.arange(mem_len), "t -> b t", b=B)
        
        q = _apply_rope(q, positions=q_positions)
        q *= head_dim**-0.5
        k = _apply_rope(k, positions=k_positions)
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)

        logits = jnp.einsum(
            "BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32
        )
        attn_mask = mem_mask[:, None, None, None, :]  # (B, 1, 1, 1, S)
        masked_logits = jnp.where(attn_mask, logits, -2.3819763e38)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out_einsum = lora.Einsum(
            shape=(num_heads, head_dim, width),
            name="out_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
        )
        return out_einsum("BTNH,NHD->BTD", encoded)


@at.typecheck
class HistoryBlock(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    integration_type: str | None = None
    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        mem_seq,
        mem_mask,
        robottt,
        deterministic=True,
    ):  # noqa: FBT002

        if self.integration_type == "modulation":
            mem_attn = MemoryAttention(name="mem_attn")

        robottt_fast_state, robottt_inner_mask, robottt_block_position, tbptt_segment_length = (
            robottt
        )
        temporal_shape = (
            robottt_inner_mask.shape[:2]
            if robottt_inner_mask is not None and robottt_inner_mask.ndim == 3
            else None
        )
        if temporal_shape is not None:
            def flatten_temporal(value):
                return value.reshape(
                    (temporal_shape[0] * temporal_shape[1], *value.shape[2:])
                )

            xs = jax.tree.map(flatten_temporal, xs)
            kv_cache = jax.tree.map(flatten_temporal, kv_cache)
            positions = flatten_temporal(positions)
            attn_mask = flatten_temporal(attn_mask)
            adarms_cond = jax.tree.map(
                lambda value: None if value is None else flatten_temporal(value),
                adarms_cond,
            )

        xs = sharding.activation_sharding_constraint(xs)
        drop = (
            nn.Dropout(self.dropout, self.dropout_bdims)
            if self.dropout
            else lambda x, _: x
        )
        
        if self.integration_type == "expert":
            attn = Attention_with_MemoryExpert(configs=self.configs, name="attn")
        else:
            attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                name = _name("pre_attention_norm", i) if self.integration_type != "expert" else _name("pre_attention_norm", i-1)
                x, gate = RMSNorm(name=name)(
                    x, adarms_cond[i]
                )  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, next_kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, post_attn, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)

        robottt_boundaries = None
        if self.integration_type == "layer" and xs[-1] is not None:
            robottt_layer = RoboTTTLayer(name="robottt")
            if temporal_shape is None:
                xs[-1], robottt_fast_state = robottt_layer(
                    xs[-1], robottt_fast_state,
                    robottt_inner_mask, robottt_block_position,
                )
            else:
                suffix = xs[-1].reshape((*temporal_shape, *xs[-1].shape[1:]))
                segment_count = temporal_shape[1] // tbptt_segment_length

                def segment_major(value):
                    value = value.reshape(
                        (
                            temporal_shape[0],
                            segment_count,
                            tbptt_segment_length,
                            *value.shape[2:],
                        )
                    )
                    return jnp.swapaxes(value, 0, 1)

                scan_inputs = (
                    segment_major(suffix),
                    segment_major(robottt_inner_mask),
                    segment_major(robottt_block_position),
                )

                def update_segment(layer, state, inputs):
                    hidden_slots, mask_slots, position_slots = inputs
                    state = jax.tree.map(jax.lax.stop_gradient, state)
                    boundary = state
                    outputs = []
                    for slot_index in range(tbptt_segment_length):
                        hidden, state = layer(
                            hidden_slots[:, slot_index],
                            state,
                            mask_slots[:, slot_index],
                            position_slots[:, slot_index],
                        )
                        outputs.append(hidden)
                    state = jax.tree.map(jax.lax.stop_gradient, state)
                    return state, (jnp.stack(outputs, axis=1), boundary)

                robottt_fast_state, (suffix, robottt_boundaries) = nn.scan(
                    update_segment,
                    variable_broadcast="params",
                    split_rngs={"params": False},
                    in_axes=0,
                    out_axes=0,
                    length=segment_count,
                )(
                    robottt_layer,
                    robottt_fast_state,
                    scan_inputs,
                )
                suffix = jnp.swapaxes(suffix, 0, 1).reshape(
                    (temporal_shape[0], temporal_shape[1], *suffix.shape[3:])
                )
                xs[-1] = flatten_temporal(suffix)
            robottt_fast_state = sharding.activation_sharding_constraint(robottt_fast_state)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # Add Memory Modulation before FFN
                if i == len(xs) - 1 and self.integration_type == "modulation":
                    mem_mod_vec = mem_attn(x, mem_seq[-1], mem_mask[-1])
                    x = MemoryRMSNorm(name="mem_rms_norm_ffn")(x, mem_mod_vec)  
                
                name=_name("pre_ffw_norm", i) if self.integration_type != "expert" else _name("pre_ffw_norm", i-1)
                x, gate = RMSNorm(name=name)(
                    x, adarms_cond[i]
                )  # noqa: PLW2901
                                
                name = _name("mlp", i) if self.integration_type != "expert" else _name("mlp", i-1)
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=name,
                    lora_config=config.lora_configs.get("ffn"),
                )(x)

            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, out, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)
        if temporal_shape is not None:
            xs = jax.tree.map(
                lambda value: value.reshape((*temporal_shape, *value.shape[1:])),
                xs,
            )
        
        return xs, (
            None if temporal_shape is not None else next_kv_cache,
            robottt_fast_state, robottt_boundaries,
        )


KVCache: TypeAlias = tuple[jax.Array, jax.Array]


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False
    
    integration_type: str | None = None
    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        embed_dim = self.configs[0].width if self.integration_type != "expert" else self.configs[1].width
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=embed_dim,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            HistoryBlock,
            prevent_cse=False,
            static_argnums=(8,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                (0, nn.broadcast, nn.broadcast, nn.broadcast),
                nn.broadcast,
            ),
            length=self.configs[0].depth,
            unroll=6,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            integration_type=self.integration_type,
        )
        self.final_norms = [
            RMSNorm(name=_name("final_norm", i) if self.integration_type != "expert" else _name("final_norm", i-1)) for i in range(len(self.configs))
        ]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "*b _t _d"] | None],
        positions: at.Int[at.Array, "*b t"],
        mask: at.Bool[at.Array, "*b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "*b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        mem_seq: Sequence[at.Float[at.Array, "b lmem _d"] | None] | None = None,
        mem_mask: Sequence[at.Bool[at.Array, "b lmem"] | None] | None = None,
        robottt=None,
        deterministic: bool = True,
    ):
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[..., None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        if robottt is None:
            robottt = (None, None, None, 1)
        elif len(robottt) == 3:
            robottt = (*robottt, 1)
        temporal_robottt = robottt[1] is not None and robottt[1].ndim == 3

        embedded, (kv_cache, robottt_fast_state, robottt_boundaries) = self.layers(
            embedded,
            kv_cache,
            positions,
            mask,
            adarms_cond,
            mem_seq,
            mem_mask,
            robottt,
            deterministic,
        )

        assert all(
            e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None
        )

        outputs = []
        for norm, hidden, cond in zip(
            self.final_norms, embedded, adarms_cond, strict=True
        ):
            if hidden is None:
                outputs.append(None)
                continue
            if hidden.ndim == 4:
                batch_size, temporal_blocks = hidden.shape[:2]
                hidden = hidden.reshape(
                    (batch_size * temporal_blocks, *hidden.shape[2:])
                )
                if cond is not None:
                    cond = cond.reshape(
                        (batch_size * temporal_blocks, *cond.shape[2:])
                    )
                hidden = norm(hidden, cond)[0]
                hidden = hidden.reshape(
                    (batch_size, temporal_blocks, *hidden.shape[1:])
                )
            else:
                hidden = norm(hidden, cond)[0]
            outputs.append(hidden)
        if self.integration_type != "layer":
            return outputs, kv_cache
        result = outputs, kv_cache, robottt_fast_state
        return (*result, robottt_boundaries) if temporal_robottt else result

    def init(self, use_adarms: Sequence[bool], mem_mods: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
            mem_seq=[
                jnp.zeros((1, 4, c.width)) if m else None
                for c, m in zip(self.configs, mem_mods, strict=True)
            ],
            mem_mask=[jnp.ones((1, 4), dtype=bool) if m else None for m in mem_mods],
            robottt=(
                create_fast_state(1),
                jnp.ones((1, 1), dtype=bool),
                jnp.zeros((1,), dtype=jnp.int32),
                1,
            ) if self.integration_type == "layer" else None,
        )
