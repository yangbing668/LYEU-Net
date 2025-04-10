import torch

from utils.averageMeter import AverageMeter


class CallBacks:
    def __init__(self, best, save_path):
        self.best = best
        self.earlyStop = AverageMeter()
        self.save_path = save_path

    def saveBestModel(self, cur, model, name):
        if cur > self.best:
            self.best = cur
            torch.save(model.state_dict(), '{}/model_best_{}.pth'.format(self.save_path, name))
            self.earlyStop.reset()
            print("\n Saving Best Model....\n")
        return

    def earlyStoping(self, cur, maxVal):
        if cur < self.best:
            self.earlyStop.update(1)

        return self.earlyStop.count > maxVal
