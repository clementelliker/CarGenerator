import torch
import torch.nn.functional as F

def DiffAugment(x, label, policy='color,translation,cutout,flip', channels_first = True, color = False):
    initlab = label
    if not channels_first:
        x = x.permute(0, 3, 1, 2)
    for p in policy.split(','):
        for f in AUGMENT_FNS[p]:
            x, label = f(x, label)
    if not channels_first:
        x = x.permute(0, 2, 3, 1)
    x = x.contiguous()
    
    return x, (label if color else initlab)

def rand_brightness(x, label):
    factor = (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)*0.5 - 0.25)
    change = torch.unsqueeze(factor.reshape(x.size(0)), dim = -1)
    #x = x + factor.repeat(1,3,x.shape[2],x.shape[3])*(torch.mean(x, dim = 1, keepdim = True) != 1.)
    x = x + factor.repeat(1,3,x.shape[2],x.shape[3])
    return x, (label + change)

def rand_saturation(x, label):
    x_mean = x.mean(dim=1, keepdim=True)
    factor = (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)*2)
    change = torch.unsqueeze(factor.reshape(x.size(0)), dim = -1)
    x = (x - x_mean) * factor + x_mean
    return x, label*change

def rand_translation(x, label, ratio=0.05):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0], value = 1.0)
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x, label

def rand_cutout(x, label, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x, label

def rand_flip(x, label, proba = 0.5):
    flips = torch.rand(x.size(0),1,1,1, dtype=x.dtype, device=x.device).repeat(1,3,x.shape[2], x.shape[3])
    x = x*(flips <= proba) + torch.flip(x, dims = [3])*(flips > proba)
    
    return x, label

AUGMENT_FNS = {
    'color': [rand_brightness],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
}