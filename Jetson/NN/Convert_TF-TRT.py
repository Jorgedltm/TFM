from tensorflow.python.compiler.tensorrt import trt_convert as trt

print('Converting to TF-TRT FP32...')

converter = trt.TrtGraphConverterV2(input_saved_model_dir='models/unet-vae-64-mse-diff/saved_model',\
                                   precision_mode=trt.TrtPrecisionMode.FP32,\
                                    max_workspace_size_bytes=8000000000)
converter.convert()
converter.save(output_saved_model_dir='models/unet-vae-64-mse-diff/saved_model_TFTRT_FP32')
print('Done Converting to TF-TRT FP32')
