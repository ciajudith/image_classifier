import time

import streamlit as st
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class StreamlitLiveMetricsCallback(Callback):

    def __init__(self, total_epochs: int):
        super().__init__()
        self.steps_per_epoch = None
        self.epoch_start_time = None
        self.last_batch_time = None
        self.total_epochs = total_epochs
        self.epoch_placeholder = st.empty()
        self.batch_bar = st.progress(0)
        self.text_placeholder = st.empty()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.last_batch_time = self.epoch_start_time
        self.steps_per_epoch = self.params.get('steps', None)

        # Titre de l'époque
        self.epoch_placeholder.markdown(f"**Époque {epoch + 1}/{self.total_epochs}**")
        self.batch_bar.progress(0)
        self.text_placeholder.text("")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        steps = self.steps_per_epoch
        if steps:
            # durée du batch
            now = time.time()
            batch_duration = now - self.last_batch_time
            self.last_batch_time = now

            # progression batch
            self.batch_bar.progress((batch + 1) / steps)

            # récupère les métriques
            acc = logs.get('accuracy', 0.0)
            prec = logs.get('precision', 0.0)
            rec = logs.get('recall', 0.0)
            loss = logs.get('loss', 0.0)

            # affichage
            self.text_placeholder.text(
                f"{batch + 1}/{steps} – {batch_duration:.0f}s/step – "
                f"accuracy: {acc:.4f} – precision: {prec:.4f} – recall: {rec:.4f} – "
                f"loss: {loss:.4f}"
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        steps = self.steps_per_epoch or 0
        elapsed = time.time() - self.epoch_start_time
        avg_step = elapsed / steps if steps else 0.0

        # récupération des métriques de fin d'époque
        acc = logs.get('accuracy', 0.0)
        prec = logs.get('precision', 0.0)
        rec = logs.get('recall', 0.0)
        loss = logs.get('loss', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)
        val_prec = logs.get('val_precision', 0.0)
        val_rec = logs.get('val_recall', 0.0)
        val_loss = logs.get('val_loss', 0.0)

        # learning rate
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        lr = float(tf.keras.backend.get_value(lr))

        self.text_placeholder.markdown(
            f"**Fin d’époque {epoch + 1}/{self.total_epochs}**  \n"
            f"{steps}/{steps} – {int(elapsed)}s – {avg_step:.0f}s/step  \n"
            f"accuracy: {acc:.4f} – precision: {prec:.4f} – recall: {rec:.4f} – loss: {loss:.4f}  \n"
            f"val_accuracy: {val_acc:.4f} – val_precision: {val_prec:.4f} – "
            f"val_recall: {val_rec:.4f} – val_loss: {val_loss:.4f} – lr: {lr:.4f}"
        )
