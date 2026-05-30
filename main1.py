import dataset
import metrics
import train
import evaluate
import model_compression

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json

##### CHANGE TO YOUR DATASET PATH #####
dataset_root_path = "YOUR_DATASET_ROOT_PATH"

SEED = 45
EPOCHS = 200
BATCH_SIZE = 32
N_LAYERS_UNFREEZE = 13
LR = 0.0001
tf.keras.utils.set_random_seed(SEED)
plt.rcParams["figure.dpi"] = 300
objects = ['cable', 'bottle', 'capsule', 'hazelnut', 'metal_nut', 
               'pill', 'screw', 'toothbrush','transistor', 'zipper', 
               'carpet', 'grid', 'leather', 'tile', 'wood']


def create_augmentation(seed=45):
    return tf.keras.Sequential([
               tf.keras.layers.RandomBrightness(factor= 0.3, seed=seed),
               tf.keras.layers.RandomContrast(factor= 0.5, seed=seed),
               tf.keras.layers.RandomRotation(factor=0.5, seed=seed),
               tf.keras.layers.GaussianNoise(stddev=0.05)
    ])


def main(seed=45):

    objects_metrics = {}
    for obj in objects:
        metrics = {}
        train_data, val_data, test_data = dataset.preprocess_dataset(dataset_root_path, obj, seed=seed)
        X_test, y_test = test_data

        # TEACHER
        print(f'*** TEACHER {obj} ***')
        teacher, history_teacher = train.train_model(model_type= 'teacher',
                                                     train_data= train_data,
                                                     val_data= val_data,
                                                     epochs=EPOCHS,
                                                     learning_rate= LR,
                                                     batch_size= BATCH_SIZE,
                                                     n_layers_unfreeze= N_LAYERS_UNFREEZE,
                                                     augmentation= create_augmentation(seed=SEED))
        teacher.save_weights(f'teacher_{obj}.weights.h5')
        history_teacher = pd.DataFrame(history_teacher.history)
        history_teacher.to_csv(f'teacher_{obj}.csv', index=False)
        teacher_converter = tf.lite.TFLiteConverter.from_keras_model(teacher)
        teacher_tflite = teacher_converter.convert()

        metrics_teacher = evaluate.eval_tflite(teacher_tflite, X_test.astype(np.float32), y_test, BATCH_SIZE)

        # Save the TFLite model and get the models storage size in MB
        with open(f"teacher_{obj}.tflite", "wb") as f:
            f.write(teacher_tflite)
        teacher_size_mb = os.path.getsize(f'teacher_{obj}.tflite') / (1024 * 1024)
        metrics_teacher.append(teacher_size_mb)

        metrics['teacher'] = metrics_teacher

        print(f'*** STUDENT FINE-TUNING {obj} ***')
        student_fine, history_stud_fine = train.train_model(model_type= 'student',
                                                            train_data= train_data,
                                                            val_data= val_data,
                                                            epochs=EPOCHS,
                                                            learning_rate= LR,
                                                            batch_size= BATCH_SIZE,
                                                            n_layers_unfreeze= N_LAYERS_UNFREEZE,
                                                            augmentation= create_augmentation(seed=SEED))
        student_fine.save_weights(f'student_fine_{obj}.weights.h5')
        history_stud_fine = pd.DataFrame(history_stud_fine.history)
        history_stud_fine.to_csv(f'student_fine_{obj}.csv', index=False)

        student_fine_converter = tf.lite.TFLiteConverter.from_keras_model(student_fine)
        student_fine_tflite = student_fine_converter.convert()

        metrics_stud_fine = evaluate.eval_tflite(student_fine_tflite, X_test.astype(np.float32), y_test, BATCH_SIZE)

        with open(f"student_fine_{obj}.tflite", "wb") as f:
            f.write(student_fine_tflite)
        student_fine_size_mb = os.path.getsize(f'student_fine_{obj}.tflite') / (1024 * 1024)
        metrics_stud_fine.append(student_fine_size_mb)

        metrics['student_fine'] = metrics_stud_fine

        # DISTILLATION
        print(f'*** DISTILLATION {obj} ***')
        student_model = train.create_student_model_mobile(shape= train_data[0][0].shape, 
                                                          augmentation= create_augmentation(seed=SEED),
                                                          n_last_layers_unfreeze= N_LAYERS_UNFREEZE)
        distiller, history_distiller = model_compression.train_distiller(teacher_trained= teacher,
                                                                         student_model= student_model,
                                                                         train_data= train_data,
                                                                         val_data= val_data,
                                                                         epochs=EPOCHS,
                                                                         learning_rate= LR,
                                                                         alpha= 0.8,
                                                                         temperature=1.7,
                                                                         batch_size= BATCH_SIZE)
        distiller.student.save_weights(f'student_distilled_{obj}.weights.h5')
        history_distiller = pd.DataFrame(history_distiller.history)
        history_distiller.to_csv(f'student_distilled_{obj}.csv', index=False)

        distilled_converter = tf.lite.TFLiteConverter.from_keras_model(distiller.student)
        distilled_tflite = distilled_converter.convert()

        metrics_distilled = evaluate.eval_tflite(distilled_tflite, X_test.astype(np.float32), y_test, BATCH_SIZE)

        with open(f"distilled_{obj}.tflite", "wb") as f:
            f.write(distilled_tflite)
        distilled_size_mb = os.path.getsize(f'distilled_{obj}.tflite') / (1024 * 1024)
        metrics_distilled.append(distilled_size_mb)

        metrics['distilled'] = metrics_distilled

        print(f'*** QUANTIZATION {obj} ***')
        distilled_student = distiller.student
        quantized = model_compression.quantize_model(distilled_student, dtype= tf.float16)
        metrics_quantized = evaluate.eval_tflite(quantized, X_test.astype(np.float32), y_test, BATCH_SIZE)
        with open(f"quantized_{obj}.tflite", "wb") as f:
            f.write(quantized)

        quantized_size_mb = os.path.getsize(f'quantized_{obj}.tflite') / (1024 * 1024)
        metrics_quantized.append(quantized_size_mb)
        metrics['quantized'] = metrics_quantized
        objects_metrics[obj] = metrics

        evaluate.plot_metrics(metrics, obj)

    with open('objects_metrics.json', 'w', encoding='utf-8') as arquivo:
        json.dump(objects_metrics, arquivo, indent=4)


if __name__ == "__main__":
    main(SEED)

