# xray-pneumonia-transfer-learning-edge-tpu

test on devboard: 
```sh
python3 classify_image.py \
  --model xray_mobilenetv2_quant_model_edgetpu.tflite  \
  --labels xray_labels.txt \
  --input chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg
```

Results:
```
mendel@green-bunny:~/xray-pneumonia-transfer-learning-edge-tpu$ python3 classify_image.py \
>   --model xray_mobilenetv2_quant_model_edgetpu.tflite  \
>   --labels xray_labels.txt \
>   --input chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg
----INFERENCE TIME----
Note: The first inference on Edge TPU is slow because it includes loading the model into Edge TPU memory.
115.7ms
14.6ms
14.6ms
14.5ms
14.4ms
-------RESULTS--------
PNEUMONIA: 7.16323
```

Model Info:
MobileNetV2 model retrained & finetuned on pneumonia data (10 epochs first layer, 5 epochs 100 layers)
Model (validation) accuracy: 97.83%
Recall: 99.83%
