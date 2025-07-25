import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.prec = tf.keras.metrics.Precision()
        self.rec  = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.prec.update_state(y_true, y_pred, sample_weight)
        self.rec.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.prec.result()
        r = self.rec.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        self.prec.reset_states()
        self.rec.reset_states()
