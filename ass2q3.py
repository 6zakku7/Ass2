import cv2
import numpy as np

confThreshold = 0.8
cam = cv2.VideoCapture(0)
classesFile = 'coco80.names'
classes = []
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNetFromDarknet('yolov3-608.cfg', 'yolov3-608.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#prices
fruit_prices = {
    'apple': 1.0,
    'banana': 0.5,
    'orange': 0.75
}

while True:
    success, img = cam.read()
    height, width, ch = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    output_layers_names = net.getUnconnectedOutLayersNames()
    LayerOutputs = net.forward(output_layers_names)
    bboxes = []
    confidences = []
    class_ids = []

    for output in LayerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(bboxes), 3))
    fruit_count = {fruit: 0 for fruit in fruit_prices.keys()}
    total_price = 0.0
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = bboxes[i]
            label = str(classes[class_ids[i]])
            confidence = round(confidences[i], 2)
            if label in fruit_prices:
                fruit_count[label] += 1
                total_price += fruit_prices[label]
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} [{confidence}]",
                            (x, y - 10), font, 1, (255, 0, 255), 2)
    total_fruits = sum(fruit_count.values())
    total_fruits_text = f"Total Fruits: {total_fruits}"
    cv2.putText(img, total_fruits_text, (10, 30), font, 1, (255, 0, 255), 2)
    total_price_text = f"Total Price: ${total_price:.2f}"
    cv2.putText(img, total_price_text, (450, 30), font, 1, (255, 0, 255), 2)
    y_offset = 60
    for fruit, count in fruit_count.items():
        fruit_text = f"{fruit.capitalize()}: {count} (${count * fruit_prices[fruit]:.2f})"
        cv2.putText(img, fruit_text, (450, y_offset), font, 1, (255, 0, 255), 2)
        y_offset += 30
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == 27:
        break
cam.release()
cv2.destroyAllWindows()