import tensorflow as tf
from metrics import BalancedAccuracy
from tensorflow.keras import ops


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_metric = BalancedAccuracy()
        self.teacher = teacher
        self.student = student

    def compile(self,
                optimizer,
                student_loss_fn,
                distillation_loss_fn,
                alpha=0.5,
                temperature=3):

        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, x=None, y=None, y_pred=None, 
                     sample_weight=None, allow_empty=False):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            ops.softmax(teacher_pred / self.temperature, axis=1), 
            ops.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        self.loss_tracker.update_state(loss)
        self.acc_metric.update_state(y, y_pred)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_metric]

    def call(self, x):
        return self.student(x)



def train_distiller(teacher_trained, 
                    student_model,
                    train_data,
                    val_data,
                    epochs,
                    learning_rate,
                    alpha,
                    temperature,
                    batch_size):
    """
    Train student model with distillation, EarlyStopping and Adam optimizer.

    Args:
       teacher_trained: teacher model pre-trained.
       student_model: lightweight model to be trained.
       train_data: tuple with training images and labels.
       val_data: tuple with validation images and labels.
       epochs: int. Number of epochs to train.
       learning_rate: float. Optimizer learning rate.
       alpha: float between 0.0 and 1.0. Controls the percentage of the student_loss and the distillation_loss in the final loss.
       temperature: float greater than or equal to 1.0. The greater results in a more softer probability distribution.
       batch_size: int. Number of images in each batch.

    Returns:
        model: distilled model
        history: distillation training history
   """

    X_train, y_train = train_data
    X_val, y_val = val_data
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    distiller = Distiller(student= student_model, teacher= teacher_trained)
    distiller.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= learning_rate),
                      student_loss_fn= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      distillation_loss_fn= tf.keras.losses.KLDivergence(),
                      alpha= alpha,
                      temperature= temperature)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.001,
                                                patience=20,
                                                restore_best_weights=True)

    history = distiller.fit(ds_train, 
                            epochs= epochs,
                            callbacks=[callback],
                            validation_data=ds_val,
                            batch_size= batch_size)

    return distiller, history


def quantize_model(model, dtype= tf.float16):
    """
    Apply quantization and convert to a TFLite model.

    Args:
        model: Full numerical precision Keras model.
        dtype: dtype to apply quantization.

    Returns:
        TFLite model quantized.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [dtype]
    converter.experimental_enable_resource_variables = True
    quantized_model = converter.convert()
    return quantized_model

