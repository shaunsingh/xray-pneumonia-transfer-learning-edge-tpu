# run on dev board
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.utils import dataset_utils
from PIL import Image

labels = dataset_utils.read_label_file('xray_labels.txt')
engine = ClassificationEngine('xray_mobilenetv2_quant_model_edgetpu.tflite')

# Run inference.
img = Image.open('chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg')
for result in engine.classify_with_image(img, top_k=3):
 print('---------------------------')
 print(labels[result[0]])
 print('Score : ', result[1])

img = Image.open('chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg')
for result in engine.classify_with_image(img, top_k=3):
 print('---------------------------')
 print(labels[result[0]])
 print('Score : ', result[1])