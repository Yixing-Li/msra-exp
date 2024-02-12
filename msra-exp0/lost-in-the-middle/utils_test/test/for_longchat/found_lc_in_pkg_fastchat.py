
class LongChatAdapter(BaseModelAdapter):
    "Model adapter for LongChat models (e.g., lmsys/longchat-7b-16k)."

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "longchat" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")

        # Apply monkey patch, TODO(Dacheng): Add flash attention support
        config = AutoConfig.from_pretrained(model_path, revision=revision)
        replace_llama_with_condense(config.rope_scaling["factor"])

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("vicuna_v1.1")



def load_model(
    model_path: str,
    device: str = "cuda",
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    gptq_config: Optional[GptqConfig] = None,
    awq_config: Optional[AWQConfig] = None,
    exllama_config: Optional[ExllamaConfig] = None,
    xft_config: Optional[XftConfig] = None,
    revision: str = "main",
    debug: bool = False,
):
    """Load a model from Hugging Face."""
    # get model adapter
    adapter = get_model_adapter(model_path)

    # Handle device mapping
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
        device, load_8bit, cpu_offloading
    )
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
        if CPU_ISA in ["avx512_bf16", "amx"]:
            try:
                import intel_extension_for_pytorch as ipex

                kwargs = {"torch_dtype": torch.bfloat16}
            except ImportError:
                warnings.warn(
                    "Intel Extension for PyTorch is not installed, it can be installed to accelerate cpu inference"
                )
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    elif device == "xpu":
        kwargs = {"torch_dtype": torch.bfloat16}
        # Try to load ipex, while it looks unused, it links into torch for xpu support
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError:
            warnings.warn(
                "Intel Extension for PyTorch is not installed, but is required for xpu inference."
            )
    elif device == "npu":
        kwargs = {"torch_dtype": torch.float16}
        # Try to load ipex, while it looks unused, it links into torch for xpu support
        try:
            import torch_npu
        except ImportError:
            warnings.warn("Ascend Extension for PyTorch is not installed.")
    else:
        raise ValueError(f"Invalid device: {device}")

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig

        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = (
                str(math.floor(psutil.virtual_memory().available / 2**20)) + "Mib"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=cpu_offloading
        )
        kwargs["load_in_8bit"] = load_8bit
    elif load_8bit:
        if num_gpus != 1:
            warnings.warn(
                "8-bit quantization is not supported for multi-gpu inference."
            )
        else:
            model, tokenizer = adapter.load_compress_model(
                model_path=model_path,
                device=device,
                torch_dtype=kwargs["torch_dtype"],
                revision=revision,
            )
            if debug:
                print(model)
            return model, tokenizer
    elif awq_config and awq_config.wbits < 16:
        assert (
            awq_config.wbits == 4
        ), "Currently we only support 4-bit inference for AWQ."
        model, tokenizer = load_awq_quantized(model_path, awq_config, device)
        if num_gpus != 1:
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=kwargs["max_memory"],
                no_split_module_classes=[
                    "OPTDecoderLayer",
                    "LlamaDecoderLayer",
                    "BloomBlock",
                    "MPTBlock",
                    "DecoderLayer",
                ],
            )
            model = accelerate.dispatch_model(
                model, device_map=device_map, offload_buffers=True
            )
        else:
            model.to(device)
        return model, tokenizer
    elif gptq_config and gptq_config.wbits < 16:
        model, tokenizer = load_gptq_quantized(model_path, gptq_config)
        if num_gpus != 1:
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=kwargs["max_memory"],
                no_split_module_classes=["LlamaDecoderLayer"],
            )
            model = accelerate.dispatch_model(
                model, device_map=device_map, offload_buffers=True
            )
        else:
            model.to(device)
        return model, tokenizer
    elif exllama_config:
        model, tokenizer = load_exllama_model(model_path, exllama_config)
        return model, tokenizer
    elif xft_config:
        model, tokenizer = load_xft_model(model_path, xft_config)
        return model, tokenizer
    kwargs["revision"] = revision

    if dtype is not None:  # Overwrite dtype if it is provided in the arguments.
        kwargs["torch_dtype"] = dtype

    # Load model
    model, tokenizer = adapter.load_model(model_path, kwargs)

    if (
        device == "cpu"
        and kwargs["torch_dtype"] is torch.bfloat16
        and CPU_ISA is not None
    ):
        model = ipex.optimize(model, dtype=kwargs["torch_dtype"])

    if (device == "cuda" and num_gpus == 1 and not cpu_offloading) or device in (
        "mps",
        "xpu",
        "npu",
    ):
        model.to(device)

    if device == "xpu":
        model = torch.xpu.optimize(model, dtype=kwargs["torch_dtype"], inplace=True)

    if debug:
        print(model)

    return model, tokenizer



@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")
