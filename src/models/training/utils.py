from torch import nn
from peft import get_peft_model, LoraConfig
from src.models.model.corigami_model import ConvTransModelSmall
from src.models.model.chiron_model import Chiron3D


def replace_bn_with_groupnorm(model):
    """
    Recursively replace all BatchNorm layers with GroupNorm(1, C)
    """
    for name, module in model.named_children():
        if "borzoi" in name:
            continue
        if isinstance(module, nn.BatchNorm1d):
            C = module.num_features
            gn = nn.GroupNorm(num_groups=1, num_channels=C)
            setattr(model, name, gn)
        elif isinstance(module, nn.BatchNorm2d):
            C = module.num_features
            gn = nn.GroupNorm(num_groups=1, num_channels=C)
            setattr(model, name, gn)
        else:
            replace_bn_with_groupnorm(module)


def get_learnable_params(model, weight_decay=1e-5):
    # Make LoRA LR = head LR = 5e-4
    adapter_lr = 5e-4
    no_decay = []
    high_lr  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Biases, LayerNorm/LayerScale weights (i.e. 1D tensors) → no weight decay
        if len(param.shape) == 1:
            no_decay.append(param)
        else:
            # Everything else (including "lora" modules) → high_lr group
            high_lr.append(param)

    return [
        {'params': high_lr, 'weight_decay': 1e-5, 'lr': adapter_lr},
        {'params': no_decay, 'weight_decay': 0, 'lr': adapter_lr},
    ]

def set_lora(model):
    lora_config = LoraConfig(
        target_modules=r"^borzoi\.(?!separable\d+).*conv_layer|^borzoi\..*to_q|^borzoi\..*to_v|^borzoi\.transformer\.\d+\.1\.fn\.1|^borzoi\.transformer\.\d+\.1\.fn\.4"
    )
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        print(f"Layer: {name} with grads {param.requires_grad}")
        if "borzoi" in name:
            continue
        else:
            param.requires_grad = True # This sets the head params to training.
            print(f"[DEBUG] Setting grad to true for layer: {name}")
    return model


def get_model(args):
    if args.borzoi:
        model = Chiron3D(mid_hidden=128, local=args.local,
                         resolution=args.resolution, n_bins=args.n_bins)
        replace_bn_with_groupnorm(model)
        model = set_lora(model)
    else:
        model = ConvTransModelSmall(mid_hidden=128, num_genomic_features=args.num_genom_feat)

    return model

