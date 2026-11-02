"""Accuracy metrics for the training modules.

Kept separate from ``utils/helpers.py`` because this module pulls in torchmetrics.
"""

from torchmetrics.classification import MulticlassAccuracy


class PercentageAccuracy(MulticlassAccuracy):
    """Micro-averaged accuracy reported as a percentage (0..100).

    The percentage convention is deliberate: ``repos_log/goal.md`` and the ``RESULT``
    block printed by ``*/train_final.py`` both state the chance threshold as a
    percentage (``1/64 = 1.56 %``), so a metric that already speaks percentages
    cannot be misread against that threshold.

    **Usage requirement:** update it and then log the *Metric object*, not the value
    it returns::

        self.acc.update(logits, labels)
        self.log("acc", self.acc, on_step=False, on_epoch=True)

    Lightning ``compute()``s a logged Metric once at epoch end (and resets it), which
    yields the exact global accuracy. Verified in-container on unequal batches
    (8/8/4 with an all/all-none/all correctness pattern): logging the Metric gave
    ``60.0`` -- the true global accuracy -- while logging the tensor returned by
    ``self.acc(logits, labels)`` gave ``6000.0``, because Lightning mean-reduces a
    plain tensor per batch and torchmetrics' ``forward`` already calls ``compute``.
    Scaling therefore happens in exactly one place: :meth:`compute`.
    """

    def __init__(self, num_classes, **kwargs):
        kwargs.setdefault("average", "micro")
        kwargs.setdefault("validate_args", False)
        super().__init__(num_classes=num_classes, **kwargs)

    def compute(self):
        return 100.0 * super().compute()
