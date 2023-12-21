import torch

class Metrics:

    @staticmethod
    def F1(y, y_hat):
        tp = torch.sum(y & y_hat)
        fp = torch.sum(~y & y_hat)
        tn = torch.sum(~y & ~y_hat)
        fn = torch.sum(y & ~y_hat)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1