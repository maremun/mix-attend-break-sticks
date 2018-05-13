import os, shutil
import torch
# from torch.autograd import Variable

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        if torch.cuda.is_available():
            return torch.tensor(h.detach(), requires_grad=True).cuda()
        else:
            return torch.tensor(h.detach(), requires_grad=True)
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz, is_cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    print("Size of generated data = {}".format(data.size()))
    if is_cuda:
        data = data.cuda()
    return data

def get_batch(source, i, seq_len, evaluation=False):
    seq_len = min(seq_len, len(source) - 1 - i)
    # data = Variable(source[i:i+seq_len], volatile=evaluation)
    with torch.set_grad_enabled(not evaluation): 
        data = source[i:i+seq_len]
    # target = Variable(source[i+1:i+1+seq_len].view(-1))
    # target = Variable(source[i+1:i+1+seq_len])
    target = source[i+1:i+1+seq_len]
    return data, target

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(model, optimizer, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
