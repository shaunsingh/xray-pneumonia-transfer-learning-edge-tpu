# xray-pneumonia-transfer-learning-edge-tpu

test on devboard: 
```sh
python3 classify_image.py \
  --model xray_mobilenetv2_quant_model_edgetpu.tflite  \
  --labels xray_labels.txt \
  --input chest_xray/val/NORMAL/NORMAL2-IM-1430-0001.jpeg
  
python3 classify_image.py \
  --model xray_mobilenetv2_quant_model_edgetpu.tflite  \
  --labels xray_labels.txt \
  --input chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg
```
