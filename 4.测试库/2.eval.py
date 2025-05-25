import argparse
import os

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
from tqdm import tqdm
import pickle
import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import os


sys.path.insert(0, '.')
sys.path.append('../')

from torch.utils.data import Dataset, DataLoader
import shutil
from sklearn.metrics import roc_curve, auc, accuracy_score
from mtcnn.detector import get_model


class IdentityMapping(nn.Module):
    def __init__(self, base):
        super(IdentityMapping, self).__init__()
        self.base = base

    def forward(self, x):
        x = self.base(x)
        return x


# 直接读取图片的数据集
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, dir_path, transform=None):
        """
        Args:
            image_paths (list): 包含所有图片路径的列表
            transform (callable, optional): 可选的图像变换
        """
        self.image_paths = image_paths
        self.dir_path = dir_path
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dir_path, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')  # 确保是 RGB 图像

        if self.transform:
            image = self.transform(image)

        return image, self.image_paths[idx]

model_names = ['resnet50']
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--input-size', default=224, type=int)
parser.add_argument('--feature-dim', default=512, type=int)
parser.add_argument('--load-path', default='J:/work/face/checkpoints/finetune_new.224.1020_epoch_59.pth.tar', type=str)
parser.add_argument('--bin-file', default='J:/InsightFace/data/lfw_funneled', type=str)
parser.add_argument('--test-pair', default='J:/InsightFace/data/lfw_test_pair.txt', type=str)
parser.add_argument('--feature-path', default='features_lfw.pkl', type=str)
parser.add_argument('--feature-path-landmarks', default='landmarks_lfw.pkl', type=str)
parser.add_argument('--save-pkl', default=True, type=bool, help='保存图片特征值')
parser.add_argument('--show-roc', default=True, type=bool, help='展示ROC图片')
parser.add_argument('--error-image', default=True, type=bool, help='错误图片输出到文件夹')


# 加载模型
# def load_model():
#     print("=> loading model '{}'".format(args.arch))
#     model = resnet50(feature_dim=args.feature_dim)
#     model = IdentityMapping(model)
#     model.cuda()
#     model = torch.nn.DataParallel(model).cuda()
#
#     checkpoint = torch.load(args.load_path)['state_dict']
#     model.load_state_dict(checkpoint, strict=False)
#     return model


# 图片去重
def extract_face_images(face1_face2_label_list):
    face_image_paths = []
    for face1, face2, _ in face1_face2_label_list:
        if face1 not in face_image_paths:
            face_image_paths.append(face1)
        if face2 not in face_image_paths:
            face_image_paths.append(face2)
    return face_image_paths


# 提取特征值
def extract_face_features(model, dir, face_image_names):
    # transform = transforms.Compose([
    #     transforms.Resize(args.input_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])])
    #
    # dataset = CustomImageDataset(image_paths=face_image_names, dir_path=dir, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    features_dict = {}

    err_det = []
    multi_face = []
    # 降低阈值的模型
    app2 = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'],
                        providers=['CUDAExecutionProvider'])

    for filename in tqdm(face_image_names):
        image_path = os.path.join(dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        faces = model.get(image)
        if len(faces) < 1:
            # raise ValueError("No faces detected in the image, path: {}".format(filename))
            # print("No faces detected in the image, path: {}".format(filename))
            #  可能存在检测不到的情况, 一直降低阈值找人脸
            det_thresh = 0.5
            while len(faces) < 1:
                app2.prepare(ctx_id=0, det_thresh=det_thresh)
                faces = app2.get(image)
                det_thresh -= 0.05
                if det_thresh ==0:
                    faces = np.zeros_like(features_dict[face_image_names[0]])
                    break
            err_det.append([filename, det_thresh])

        if len(faces) > 1:
            multi_face.append(filename)
            # 找特征点在图中的且面积最大那个脸
            max_area = 0
            max_face = None
            for face in faces:
                keypoints = face["kps"]
                bbox = face["bbox"]
                all_non_negative = all(x >= 0 and y >= 0 for x, y in keypoints)
                if all_non_negative:
                    area = np.abs(bbox[2] - bbox[0]) * np.abs(bbox[3] - bbox[1])
                    if area > max_area:
                        max_face = face
                        max_area = area
            faces = [max_face]
            # print("Warning: Multiple faces detected. Using first detected face,path: {}".format(filename))
        features_dict[filename] = faces[0].embedding

    with open(args.feature_path, 'wb') as f:
        pickle.dump(features_dict, f)
    with open('err_det.txt','w') as f:
        for item in err_det:
            f.write(f"{item}\n")
    with open('multi_face.txt','w') as f:
        for item in multi_face:
            f.write(f"{item}\n")
    return features_dict

# 获取两两图片余弦相似度
def get_score_label(features_dict, test_pair):
    scores = []
    labels = []

    for face1, face2, label in test_pair:
        emb1 = features_dict[face1]
        emb2 = features_dict[face2]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))  # 余弦相似度
        scores.append(similarity)
        labels.append(int(label))
    return scores, labels


# 预处理图片,获取人脸特征点
def preprocess(dir, file_names):
    image_landmarks = {}
    pnet, rnet, onet = get_model()
    for file in tqdm(file_names, desc="get face attributes"):
        file_path = os.path.join(dir, file)
        is_valid, bounding_boxes, landmarks = get_face_all_attributes(file_path, models=[pnet, rnet, onet])
        if is_valid:
            image_landmarks[file] = landmarks
    if args.save_pkl:
        with open(args.feature_path_landmarks, 'wb') as f:
            pickle.dump(image_landmarks, f)
    return image_landmarks


# 评估roc,acc,tpr,fpr,导出错误的图片
def evaluate(features_dict, test_pair):
    # 计算余弦相似度, 获取真实标签
    scores, labels = get_score_label(features_dict, test_pair)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    if args.show_roc:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    # 通过roc曲线计算最优threshold, 最大化TPR最小化FPR
    best_threshold_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_idx]

    best_result = (scores >= best_threshold)
    acc = np.mean((best_result == labels).astype(int))
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Best threshold: {best_threshold}, Accuracy: {acc:.4f}")

    # 获取TPR@FPR 0.001 0.01 0.1
    target_fprs = [1e-3, 1e-2, 1e-1]
    for target_fpr in target_fprs:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tar = tpr[idx]
        print(f"TAR@FPR={target_fpr:.0e} = {tar:.4f}")

    # 遍历所有距离算出最优的阈值
    def cal_accuracy(y_score, y_true):
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        best_acc = 0
        best_th = 0
        for i in range(len(y_score)):
            th = y_score[i]
            # 余弦相似度1相同,越大越相似
            y_test = (y_score >= th)
            acc = np.mean((y_test == y_true).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = th
        print(f"Best threshold: {best_th}，Accuracy：{best_acc}")

        return (best_acc, best_th)

    best_acc, best_th = cal_accuracy(scores, labels)

    if args.error_image:
        def get_error_list(best_th, y_score, y_true):
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            y_test = (y_score >= best_th)
            y_test = np.asarray(y_test, dtype=int)
            differences = []
            for idx, (val1, val2) in enumerate(zip(y_test, y_true)):
                if val1 != val2:
                    differences.append({
                        'index': idx, 'img1': test_pair[idx][0], 'img2': test_pair[idx][1],
                        'predict': val1, 'score': y_score[idx],
                        'trueLabel': val2
                    })
                    # 移动图片到文件夹便于查看
                    file1 = os.path.join(args.bin_file, test_pair[idx][0])
                    file2 = os.path.join(args.bin_file, test_pair[idx][1])
                    dest_file1 = os.path.join('./error_image', test_pair[idx][0].replace('/', '-'))
                    dest_file2 = os.path.join('./error_image', test_pair[idx][1].replace('/', '-'))
                    try:
                        shutil.copy(file1, dest_file1)
                        shutil.copy(file2, dest_file2)
                    except Exception as e:
                        print(f"Failed to copy ,{e}")

            import pandas as pd
            df = pd.DataFrame(differences, columns=['index', 'img1', 'img2', 'predict', 'score', 'trueLabel'])
            df.to_csv('difference.csv', index=False)
            return differences

        differences = get_error_list(best_th, scores, labels)
        print(f"错误数: {len(differences)}")


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    # 获取测试对
    with open(args.test_pair, 'r') as f:
        test_pair = [line.rstrip('\n').split(' ') for line in f]
        face_image_paths = extract_face_images(test_pair)


    # 获取图片特征值
    if not os.path.exists(args.feature_path):
        # 加载模型 模型自带人脸检测
        app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'],
                           providers=['CUDAExecutionProvider'])
        app.prepare(ctx_id=0)  # ctx_id=-1 for CPU, 0 for GPU
        features_dict = extract_face_features(app, args.bin_file, face_image_paths)
    else:
        with open(args.feature_path, 'rb') as f:
            features_dict = pickle.load(f)

    evaluate(features_dict, test_pair)
