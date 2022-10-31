# from fnmatch import fnmatchcase
# from platform import java_ver
import torch
import os
import numpy as np
import models
import cv2
import time

pretrain_mask_weight = [os.path.join(os.path.dirname(__file__), "weights", "cardiomegaly.pt"),
                        os.path.join(os.path.dirname(__file__), "weights", "pneumonia.pt"), os.path.join(os.path.dirname(__file__), "weights", "pleural_effusion.pt")]
pretrain_gender_weight = os.path.join(
    os.path.dirname(__file__), "weights", "gender_weight.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_classes = 1
Cardio_Autoencoder = models.Combined_to_oneNet(
    spurious_weight=pretrain_gender_weight, spurious_oc=2, num_verb=target_classes).to(device)
Pneumonia_Autoencoder = models.Combined_to_oneNet(
    spurious_weight=pretrain_gender_weight, spurious_oc=2, num_verb=target_classes).to(device)
Pleural_Autoencoder = models.Combined_to_oneNet(
    spurious_weight=pretrain_gender_weight, spurious_oc=2, num_verb=target_classes).to(device)
Cardio_Autoencoder.load_state_dict(torch.load(
    pretrain_mask_weight[0])["autoencoder_state_dict"])
Pneumonia_Autoencoder.load_state_dict(torch.load(
    pretrain_mask_weight[1])["autoencoder_state_dict"])
Pleural_Autoencoder.load_state_dict(torch.load(
    pretrain_mask_weight[2])["autoencoder_state_dict"])


def preprocess(img_path):
    full_img = cv2.imread(img_path)
    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
    full_img = cv2.resize(full_img, (256, 256))
    full_img = np.transpose(full_img, (2, 0, 1))/255.0
    full_img = torch.from_numpy(full_img.copy()).float()
    full_img = full_img.unsqueeze(0)
    return full_img


def validate(img_path, masked_img_dir=None):
    # if not os.path.exists(masked_img_dir):
    #     os.makedirs(masked_img_dir)
    test_image = preprocess(img_path).to(device)
    print('Start validating')
    Cardio_Autoencoder.eval()
    Pneumonia_Autoencoder.eval()
    Pleural_Autoencoder.eval()
    start_time = time.time()

    with torch.no_grad():
        cardio_task_pred, cardio_adv_pred, cardio_draw_image, cardio_task_mask = Cardio_Autoencoder(
            test_image)
        pneumo_task_pred, pneum_adv_pred, pneumo_draw_image, pneumo_mask = Pneumonia_Autoencoder(
            test_image)
        pleural_task_pred, pleural_adv_pred, pleural_draw_image, pleural_mask = Pleural_Autoencoder(
            test_image)
        cardio_task_pred = torch.squeeze(cardio_task_pred, 1)
        pneumo_task_pred = torch.squeeze(pneumo_task_pred, 1)
        pleural_task_pred = torch.squeeze(pleural_task_pred, 1)
        print('Finished Validating')

    end_time = time.time()
    duration = end_time - start_time
    print(duration)

    return cardio_task_pred, pneumo_task_pred, pleural_task_pred
