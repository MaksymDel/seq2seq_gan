import torch


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def detach_fake_batch(fake_batch):
    fake_batch['source_tokens']['onehots'] = fake_batch['source_tokens']['onehots'].detach()
    return fake_batch


def mean_of_list(l):
    return round(sum(l) / float(len(l)), 4)


def to_cuda(obj, modules=False):
    """
    Move to GPU
    """

    if modules:
        return {key: value.cuda() for key, value in obj.items()}

    if isinstance(obj, torch.Tensor):
        return obj.cuda()
    elif isinstance(obj, dict):
        return {key: to_cuda(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [to_cuda(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([to_cuda(item) for item in obj])