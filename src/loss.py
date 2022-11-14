import torch
import torch.nn as nn


def multilabel_categorical_crossentropy(y_true, y_pred):
    """Cross-entropy for multi-label classification
    Description: The shape of y_true and y_pred are the same,
    and the element of y_true is either 0 or 1.
    1 indicates that the corresponding class is the target class,
    and 0 indicates that the corresponding class is the non-target class.
    Warning: Please ensure that the range of y_pred is all real numbers,
    in other words, in general, y_pred
    No need to add activation function,
    especially not to add sigmoid or softmax! predict
    The stage outputs classes with y_pred greater than 0.
    When in doubt, read carefully and understand
    This article.
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e30
    y_pred_pos = y_pred - (1 - y_true) * 1e30
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


class balanced_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        f1 = self.f1(logits, labels)
        loss = multilabel_categorical_crossentropy(labels, logits)
        loss = loss.mean()
        return loss, f1

    def f1(self, logits, labels):
        predicted = torch.zeros_like(logits).to(logits)
        predicted[logits > 0] = 1
        tp = torch.sum(torch.logical_and(predicted, labels).float(), dim=1)
        fn_fp = torch.sum(torch.logical_xor(predicted, labels).float(), dim=1)
        epsilon = torch.zeros_like(tp) + 1e-10
        f1 = tp / (tp + (fn_fp / 2) + epsilon)
        f1 = f1.mean()
        return f1

