import cv2
import os
import configuration as config
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

'''
input: height, width
output: images[], labels[], bboxes[]
'''

USED_CLASSES = ['accordion', 'ant', 'buddha', 'camera','octopus']

def get_data(height, width):
    images = []
    labels = []
    bboxes = []

    # Lấy danh sách các file trong thư mục config.ANNOTATIONS_PATH
    csvFileNames = os.listdir(config.ANNOTATION_PATH)

    # Duyệt qua các file CSV để đọc dữ liệu
    for csvFileName in csvFileNames:
        if not csvFileName.endswith('.csv') or csvFileName.split('.')[0] not in USED_CLASSES:
            continue
        # Mở file CSV
        csvPath = config.ANNOTATION_PATH / csvFileName
        rows = pd.read_csv(str(csvPath), encoding='utf-8')

        # Đọc từng dòng
        for index, row in rows.iterrows():
            # Lấy thông tin file ảnh
            (filename, x1, y1, x2, y2, label) = row

            # Đọc file ảnh với hàm opencv
            imagePath = config.IMAGE_PATH / label / filename
            image = cv2.imread(str(imagePath))[..., ::-1]
            if image is not None:
                (h, w, depth) = image.shape

                # Chuẩn hóa về miền giá trị [0..1]
                x1 = float(x1) / w
                y1 = float(y1) / h
                x2 = float(x2) / w
                y2 = float(y2) / h

                # Load lại ảnh với hàm load_img với kích thước height, width
                image = cv2.resize(image, (width, height))

                images.append(image)
                labels.append(label)
                bboxes.append((x1, y1, x2, y2))
    return images, labels, bboxes

'''
input: images[], labels[], bboxes[]
output: (images_train,labels_train,bboxes_train), (images_valid,labels_valid,bboxes_valid)
'''
def preprocess_input(images, labels, bboxes):
    # Chuyển các dữ liệu về numpy số thực
    # Chuyển ảnh về miền giá trị [0..1]
    images = [image / 255. for image in images]

    # one-hot encoding các labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    print('Classes: ', lb.classes_)

    # Chia thành 2 tập: train và valid
    images_train, images_valid = train_test_split(images, test_size=0.2, train_size=0.8, random_state=42)
    labels_train, labels_valid = train_test_split(labels, test_size=0.2, train_size=0.8, random_state=42)
    bboxes_train, bboxes_valid = train_test_split(bboxes, test_size=0.2, train_size=0.8, random_state=42)
    return (images_train,labels_train,bboxes_train), (images_valid,labels_valid,bboxes_valid)

if __name__ == "__main__":
    images, labels, bboxes = get_data(224, 224)
    print("data: ", len(images))

    (images_train, labels_train, bboxes_train), (images_valid, labels_valid, bboxes_valid) = preprocess_input(images, labels, bboxes)
    print("Train: ", len(images_train))
    print("Valid: ", len(images_valid))

    img = images_train[0]
    img = (img * 255).astype(np.uint8)
    h, w, _ = img.shape
    bboxes = bboxes_train[0]
    x1, y1, x2, y2 = bboxes
    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)

    print(labels_train[0])

    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    plt.imshow(img)

    plt.savefig('test.jpg')