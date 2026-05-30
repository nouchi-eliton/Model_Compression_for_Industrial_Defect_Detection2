import tensorflow as tf
from metrics import BalancedAccuracy

def create_teacher_model(shape, augmentation=None, n_last_layers_unfreeze=None):
    """
    Create the teacher model (ResNet50V2).
    Args:
        shape: input shape
        augmentation: tf.keras.Sequential with transformation layers
        n_last_layers_unfreeze: number of layers to unfreeze and enable training
    Return:
        model: teacher model 
    """

    base_model = tf.keras.applications.ResNet50V2(
      include_top = False,
      weights = 'imagenet',
      input_shape = shape
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=shape)
    if augmentation is not None:
        x = augmentation(inputs)
    else:
        x = inputs
    preprocess = tf.keras.applications.resnet_v2.preprocess_input(x)
    x = base_model(preprocess, training = False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs, output)

    if n_last_layers_unfreeze is not None:
        if n_last_layers_unfreeze >= 1:
            model.layers[-1].trainable = True
        if n_last_layers_unfreeze >= 2:
            model.layers[-2].trainable = True
        if n_last_layers_unfreeze >= 3:
            for layer in base_model.layers[-(n_last_layers_unfreeze-2):]:
                layer.trainable = True

    return model


def create_student_model_mobile(shape, augmentation=None, n_last_layers_unfreeze=None):
    """
    Create the student model (MobileNetV3Small).
    Args:
        shape: input shape
        augmentation: tf.keras.Sequential with transformation layers
        n_last_layers_unfreeze: number of layers to unfreeze and enable training
    Return:
        model: student model 
    """
    base_model = tf.keras.applications.MobileNetV3Small(
      include_top = True,
      weights = 'imagenet',
      input_shape = shape
    )
    base_model.trainable = False 
    base_model = tf.keras.Model(inputs=base_model.input, 
                       outputs= base_model.get_layer(index=-2).output)

    inputs = tf.keras.Input(shape=shape)
    if augmentation is not None:
        x = augmentation(inputs)
    else:
        x = inputs
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    x = base_model(preprocess, training = False) 
    output = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs, output)

    if n_last_layers_unfreeze is not None:
        if n_last_layers_unfreeze >= 1:
            model.layers[-1].trainable = True
        if n_last_layers_unfreeze >= 2:
            for layer in base_model.layers[-(n_last_layers_unfreeze-1):]:
                layer.trainable = True

    return model



def train_model(model_type, 
                train_data,
                val_data,
                epochs,
                learning_rate, 
                batch_size, 
                n_layers_unfreeze, 
                augmentation= None):
    """
    Train teacher or student model with EarlyStopping and Adam optimizer.

    Args:
       model_type: string identifying the model type. Must be "teacher" or "student".
       train_data: tuple with training images and labels.
       val_data: tuple with validation images and labels.
       epochs: int. Number of epochs to train.
       learning_rate: float. Optimizer learning rate.
       batch_size: int. Number of images in each batch.
       n_last_layers_unfreeze: number of layers to unfreeze and enable training
       augmentation: tf.keras.Sequential with transformation layers

    Returns:
        model: trained model
        history: training history
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    ds_val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    if model_type == "teacher":
        model = create_teacher_model(shape= X_train[0].shape,
                                     augmentation= augmentation, 
                                     n_last_layers_unfreeze= n_layers_unfreeze) 
    elif model_type == 'student':
        model = create_student_model_mobile(shape= X_train[0].shape, 
                                            augmentation= augmentation,
                                            n_last_layers_unfreeze= n_layers_unfreeze)
    else:
        print("Invalid model type!")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.001,
                                                patience=20,
                                                restore_best_weights=True)

    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= learning_rate),
                          loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                          metrics= [BalancedAccuracy()])
    history = model.fit(ds_train,
                        epochs= epochs,
                        callbacks= [callback],
                        validation_data= ds_val,)

    return model, history

