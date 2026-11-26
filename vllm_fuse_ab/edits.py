EDITS = [
    # 1. import
    ("import vllm_hcu.platforms.envs as henvs\n",
     "import vllm_hcu.platforms.envs as henvs\nfrom vllm_fuse_ab import fuse_a, fuse_b\n"),
    ("""        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=vllm_config.parallel_config.enable_lightly_cp,
        )
""", """        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=not fuse_a.use_for_layer(vllm_config, config, int(prefix.split(".")[-2])),
            prefix=f"{prefix}.o_proj",
            disable_tp=vllm_config.parallel_config.enable_lightly_cp,
        )
"""),
    ("""        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
""", """        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.fuse_a = fuse_a.use_for_layer(vllm_config, config, layer_idx)
        if self.fuse_a and not (isinstance(self.self_attn, DeepseekV2MLAAttention)
                                and isinstance(self.mlp, DeepseekV2MoE)):
            raise RuntimeError(f"fuse A expects MLA + MoE at layer {layer_idx}")
        if self.fuse_a:
            fuse_a.register_moe(layer_idx, self.mlp)
        self.fuse_b = fuse_b.use_for_layer(vllm_config, config, layer_idx)
        if self.fuse_b:
            if not isinstance(self.self_attn, DeepseekV2MLAAttention):
                raise RuntimeError(f"fuse B expects MLA at layer {layer_idx}")
            fuse_b.register_attn(layer_idx, self.self_attn)
"""),
    ("""        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
""", """        if self.fuse_a:
            hidden_states = fuse_a.post_attention(self, hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
"""),
    ("""            gate=self.gate,
""", """            gate=fuse_a.moe_gate(get_current_vllm_config(), config, int(prefix.split(".")[-2]), self.gate),
"""),
    ("""    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        enable_lightly_cp = get_forward_context().enable_lightly_cp
""", """    def forward(self, hidden_states: torch.Tensor, quanted=None, topk=None) -> torch.Tensor:
        enable_lightly_cp = get_forward_context().enable_lightly_cp
"""),
    ("""        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
""", """        kw = fuse_a.moe_experts_kwargs(quanted, topk)
        if topk is not None:
            final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=None, **kw)
        elif self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states, **kw
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits, **kw
            )
"""),
    ("""            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
""", """            prefix=f"{prefix}.experts",
            reduce_results=fuse_b.moe_reduce_results(get_current_vllm_config(), config, int(prefix.split(".")[-2])),
            scoring_func=getattr(config, "scoring_func", "softmax"),
"""),
    ("""        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
""", """        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        elif self.fuse_b:
            hidden_states = fuse_b.input_layernorm(self, hidden_states, residual)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if self.fuse_b:
            attn_kwargs["fuse_b_qkv"] = True
"""),
    ("""    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states, llama_4_scaling)
""", """    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
        fuse_b_qkv: bool = False,
    ) -> torch.Tensor:
        if fuse_b_qkv:
            return fuse_b.mla_forward(self.mla_attn, positions, hidden_states, llama_4_scaling)
        return self.mla_attn(positions, hidden_states, llama_4_scaling)
"""),
    ("""    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
""", """    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb, kw=None
    ) -> torch.Tensor:
"""),
    ("""            kw, _ = self.wk_weights_proj(hidden_states)
""", """            kw = self.wk_weights_proj(hidden_states)[0] if kw is None else kw
""", 2),
    ("""            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
""", """            prefix=f"{prefix}.layers",
        )
        fuse_a.init(vllm_config, config.hidden_size, vllm_config.model_config.dtype)
        self.fuse_final = fuse_a.use_for_final(vllm_config, config, config.num_hidden_layers - 1)

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
"""),
    ("""        hidden_states, _ = self.norm(hidden_states, residual)
""", """        if self.fuse_final:
            hidden_states = torch.ops.vllm.hcu_fuse_a_rs_norm_ag(
                hidden_states, residual, self.norm.weight, self.norm.variance_epsilon, False
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
"""),
]
