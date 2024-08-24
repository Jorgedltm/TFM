from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
import numpy as np

print('Converting to TF-TRT FP32...')

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(physical_devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=2048)])

MAX_BATCH_SIZE = 3

# Crear datos de prueba sintéticos con las formas correctas
in_1 = np.random.rand(MAX_BATCH_SIZE, 144, 160, 2).astype(np.float32)
in_2 = np.random.rand(MAX_BATCH_SIZE, 2, 16).astype(np.float32)

def input_fn():
    yield [in_1, in_2]

converter = trt.TrtGraphConverterV2(input_saved_model_dir='models/unet-vae-64-mse-diff/saved_model/',\
                                    precision_mode=trt.TrtPrecisionMode.FP16,\
                                    max_workspace_size_bytes=2 << 29,\
                                    maximum_cached_engines=100,\
                                    minimum_segment_size=3)

converter.convert()
converter.build(input_fn=input_fn)
converter.save(output_saved_model_dir='models/unet-vae-64-mse-diff/saved_model_TFTRT_FP16_1GB',save_gpu_specific_engines=True)
print('Done Converting to TF-TRT FP32')
