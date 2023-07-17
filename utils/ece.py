from torchmetrics import CalibrationError, MeanSquaredError
import torch.nn.functional as F

class ECEMetricWrapper:
    def __init__(self, conf_fn, n_bins=15, ignore_id=255, num_classes=19):
        """
        n_bins (int): number of confidence interval bins
        """
        self.n_bins = n_bins
        self.compute_confidence = conf_fn
        self.ignore_id = ignore_id
        self.num_classes = num_classes
        self.metric = CalibrationError(n_bins=n_bins)

    def compute_for_dataset(self, loader):
        for data in loader:
            x = data[0]
            y = data[1]
            out = self.compute_confidence(x)
            y = y.view(-1)
            out = out.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            out = out[y != self.ignore_id, :]
            y = y[y != self.ignore_id]
            self.metric.update(out, y)
        return self.metric.compute().item()


class BrierScoreWrapper:
    def __init__(self, conf_fn, ignore_id=255, num_classes=19):
        self.compute_confidence = conf_fn
        self.ignore_id = ignore_id
        self.num_classes = num_classes
        self.metric = MeanSquaredError(compute_on_cpu=True)

    def compute_for_dataset(self, loader):
        for data in loader:
            x = data[0]
            y = data[1][:, 0]
            out = self.compute_confidence(x)
            y[y == self.ignore_id] = self.num_classes
            y_oh = F.one_hot(y, self.num_classes+1).permute(0, 3, 1, 2)[:, :self.num_classes]
            mask = y_oh.sum(1, keepdim=True).repeat(1, self.num_classes, 1, 1)
            out = out * mask # correct for ignore pixels
            self.metric.update(out, y_oh)
        return self.metric.compute().item() * self.num_classes