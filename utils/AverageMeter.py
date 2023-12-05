
class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __copy__(self):
        new_obj = AverageMeter()
        new_obj.val, new_obj.avg, new_obj.sum, new_obj.count = self.val, self.avg, self.sum, self.count
        return new_obj


class AverageMeterDict(dict):
    def __init__(self):
        super(AverageMeterDict, self).__init__()

    def state_dict(self):
        ret_dict = dict()
        for key, value in self.items():
            ret_dict[key] = [value.val, value.avg, value.sum, value.count]
        return ret_dict

    def load_state_dict(self):
        pass