def Tuner(modeld, modelg, args):
    """Fine-tuning Options"""
    modeld.requires_grad_(False)
    modelg.requires_grad_(False)  

    if args.tune_mode == "fft":
        modeld.requires_grad_(True)
        modelg.requires_grad_(True)
    elif args.tune_mode == "last":
        print("last layer")
        for n, p in modelg.named_parameters():
            if 'decoder_layer4' in n:
                p.requires_grad_(True)

    elif args.tune_mode == "ln":
        for m in modeld.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        for m in modelg.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)

    elif args.tune_mode == "bn":
        for m in modeld.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.requires_grad_(True)
        for m in modelg.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.requires_grad_(True)
    
    elif args.tune_mode == "bias":
        for n, p in modeld.named_parameters():
            if 'bias' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_parameters():
            if 'bias' in n:
                p.requires_grad_(True)

    elif args.tune_mode == "adpt":
        for n, p in modeld.named_parameters():
            if 'adapter_' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_parameters():
            if 'adapter_' in n:
                p.requires_grad_(True)
    elif args.tune_mode == "adpt_bias":
        for n, p in modeld.named_parameters():
            if 'adapter_' in n:
                p.requires_grad_(True)
            if 'bias' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_parameters():
            if 'adapter_' in n:
                p.requires_grad_(True)
            if 'bias' in n:
                p.requires_grad_(True)
    elif args.tune_mode == "adpt_last":
        for n, p in modeld.named_parameters():
            if 'adapter_' in n:
                p.requires_grad_(True)
            if 'decoder_layer4' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_parameters():
            if 'adapter_' in n:
                p.requires_grad_(True)


    elif args.tune_mode == "lora":
        for n, p in modeld.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(True)
    elif args.tune_mode == "lora_bias":
        for n, p in modeld.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(True)
            if 'bias' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(True)
            if 'bias' in n:
                p.requires_grad_(True)
    elif args.tune_mode == "lora_last":
        for n, p in modeld.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(True)
            if 'decoder_layer4' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(True)

    elif args.tune_mode in ["shallow", "deep"]:
        for n, p in modelg.named_parameters():
            if args.roi_z == 64:
                if 'prompt_embeddings1' in n:
                    p.requires_grad_(True)
                if args.deep:   
                    if 'deep_prompt_embeddings1' in n:
                        p.requires_grad_(True)
            elif args.roi_z == 32:
                if 'prompt_embeddings2' in n:
                    p.requires_grad_(True)
                if args.deep:
                    if 'deep_prompt_embeddings2' in n:
                        p.requires_grad_(True)

        for n, p in modeld.named_parameters():
            if args.roi_z==64:
                if 'prompt_embeddings1' in n:
                    p.requires_grad_(True)
            elif args.roi_z==32:
                if 'prompt_embeddings2' in n:
                    p.requires_grad_(True)

    else:
        print("None of layers are unfrozen")

    # show_param(modeld)
    # show_params(modelg)
            
    return modeld, modelg