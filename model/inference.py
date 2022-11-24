import torch
import os
import numpy as np
import models
import cv2
import time
import base64

our_weight = [os.path.join(os.path.dirname(__file__), "weights", "cardiomegaly.pt"),
              os.path.join(os.path.dirname(__file__), "weights", "pneumonia.pt"), os.path.join(os.path.dirname(__file__), "weights", "pleural_effusion.pt")]
baseline_weight = {
    "cardiomegaly": os.path.join(os.path.dirname(__file__), "weights", "cardio_baseline.pt"),
    "pneumonia": os.path.join(os.path.dirname(__file__), "weights", "pneumonia_baseline.pt"),
    "pleural_effusion": os.path.join(os.path.dirname(__file__), "weights", "pleural_baseline.pt")
}
vis_mask_weight = {
    "cardiomegaly": os.path.join(os.path.dirname(__file__), "weights", "cardiomegaly_vis.pt"),
    "pneumonia": os.path.join(os.path.dirname(__file__), "weights", "pneumonia_vis.pt"),
    "pleural_effusion": os.path.join(os.path.dirname(__file__), "weights", "pleural_effusion_vis.pt")
}
pretrain_gender_weight = os.path.join(
    os.path.dirname(__file__), "weights", "gender_weight.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_classes = 1
# our model
Cardio_Autoencoder = models.Combined_to_oneNet(
    spurious_weight=pretrain_gender_weight, spurious_oc=2, num_verb=target_classes, grad_reverse=False).to(device)
Pneumonia_Autoencoder = models.Combined_to_oneNet(
    spurious_weight=pretrain_gender_weight, spurious_oc=2, num_verb=target_classes, grad_reverse=False).to(device)
Pleural_Autoencoder = models.Combined_to_oneNet(
    spurious_weight=pretrain_gender_weight, spurious_oc=2, num_verb=target_classes, grad_reverse=False).to(device)
# baseline model
Cardio_baseline = models.target_classifier_new(oc=1).to(device)
Pneumonia_baseline = models.target_classifier_new(oc=1).to(device)
Pleural_baseline = models.target_classifier_new(oc=1).to(device)

# vis mask model
Cardio_vis_Autoencoder = models.Autoencoder_new().to(device)
Pneumonia_vis_Autoencoder = models.Autoencoder_new().to(device)
Pleural_vis_Autoencoder = models.Autoencoder_new().to(device)

# load weight
print("Loading weight...")
Cardio_Autoencoder.load_state_dict(torch.load(
    our_weight[0])["autoencoder_state_dict"])
Pneumonia_Autoencoder.load_state_dict(torch.load(
    our_weight[1])["autoencoder_state_dict"])
Pleural_Autoencoder.load_state_dict(torch.load(
    our_weight[2])["autoencoder_state_dict"])
Cardio_baseline.load_state_dict(torch.load(
    baseline_weight["cardiomegaly"])["model_state_dict"])
Pneumonia_baseline.load_state_dict(torch.load(
    baseline_weight["pneumonia"])["model_state_dict"])
Pleural_baseline.load_state_dict(torch.load(
    baseline_weight["pleural_effusion"])["model_state_dict"])
Cardio_vis_Autoencoder.load_state_dict(torch.load(
    vis_mask_weight["cardiomegaly"])["autoencoder_state_dict"])
Pneumonia_vis_Autoencoder.load_state_dict(torch.load(
    vis_mask_weight["pneumonia"])["autoencoder_state_dict"])
Pleural_vis_Autoencoder.load_state_dict(torch.load(
    vis_mask_weight["pleural_effusion"])["autoencoder_state_dict"])


def preprocess(img_path):
    full_img = cv2.imread(img_path)
    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
    full_img = cv2.resize(full_img, (256, 256))
    image_base64 = to_base64(full_img)
    full_img = np.transpose(full_img, (2, 0, 1))/255.0
    full_img = torch.from_numpy(full_img.copy()).float()
    full_img = full_img.unsqueeze(0)
    return full_img, image_base64


def postprocess(img_path, mask, mask_type="None"):
    db_root = r"C:\medex-backend\db"
    full_img = cv2.imread(img_path)
    full_img = cv2.resize(full_img, (256, 256))
    mask = (mask-mask.min())/(mask.max()-mask.min())
    mask = np.uint8((mask.squeeze(0).squeeze(0).cpu().detach()*255))
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    result = cv2.addWeighted(heatmap, 0.5, full_img, 0.5, 0)
    if not os.path.exists(os.path.join(db_root, "imgs", "{}".format(img_path.split("\\")[-1].split(".")[0]))):
        os.makedirs(os.path.join(db_root, "imgs", "{}".format(
            img_path.split("\\")[-1].split(".")[0])))
    mask_path = os.path.join(db_root, "imgs", "{}".format(
        img_path.split("\\")[-1].split(".")[0]), "{}.jpg".format(mask_type))
    cv2.imwrite(mask_path, result)
    result_base64 = to_base64(result)
    return mask_path, result_base64


def to_base64(img):
    # im_arr: image in Numpy one-dim array format.
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr).decode()
    return im_b64


def validate(img_path, masked_img_dir=None):
    # if not os.path.exists(masked_img_dir):
    #     os.makedirs(masked_img_dir)
    test_image, image_base64 = preprocess(img_path)
    test_image = test_image.to(device)
    print('Start validating')
    Cardio_Autoencoder.eval()
    Pneumonia_Autoencoder.eval()
    Pleural_Autoencoder.eval()
    Cardio_baseline.eval()
    Pneumonia_baseline.eval()
    Pleural_baseline.eval()
    Cardio_vis_Autoencoder.eval()
    Pneumonia_vis_Autoencoder.eval()
    Pleural_vis_Autoencoder.eval()
    start_time = time.time()

    with torch.no_grad():
        cardio_task_pred, cardio_draw_image, cardio_task_mask = Cardio_Autoencoder(
            test_image)
        pneumo_task_pred, pneumo_draw_image, pneumo_mask = Pneumonia_Autoencoder(
            test_image)
        pleural_task_pred, pleural_draw_image, pleural_mask = Pleural_Autoencoder(
            test_image)
        _, cardio_mask = Cardio_vis_Autoencoder(
            test_image)
        _, pneumonia_mask = Pneumonia_vis_Autoencoder(
            test_image)
        _, pleural_mask = Pleural_vis_Autoencoder(
            test_image)
        cardio_baseline_pred = Cardio_baseline(test_image)
        pneumo_baseline_pred = Pneumonia_baseline(test_image)
        pleural_baseline_pred = Pleural_baseline(test_image)

        cardio_task_pred = torch.squeeze(cardio_task_pred, 1)
        pneumo_task_pred = torch.squeeze(pneumo_task_pred, 1)
        pleural_task_pred = torch.squeeze(pleural_task_pred, 1)
        cardio_baseline_pred = torch.squeeze(cardio_baseline_pred, 1)
        pneumo_baseline_pred = torch.squeeze(pneumo_baseline_pred, 1)
        pleural_baseline_pred = torch.squeeze(pleural_baseline_pred, 1)
    cardio_vis_path, cardio_vis_base64 = postprocess(
        img_path, cardio_mask, "cardio")
    pneumonia_vis_path, pneumonia_vis_base64 = postprocess(
        img_path, pneumonia_mask, "pneumonia")
    pleural_vis_path, pleural_vis_base64 = postprocess(
        img_path, pleural_mask, "pleural")
    ret = {"image_base64": image_base64,
           "cardio": {"our": cardio_task_pred.item(), "baseline": cardio_baseline_pred.item(), "vis": cardio_vis_path, "vis_base64": cardio_vis_base64},
           "pneumonia": {"our": pneumo_task_pred.item(), "baseline": pneumo_baseline_pred.item(), "vis": pneumonia_vis_path, "vis_base64": pneumonia_vis_base64},
           "pleural": {"our": pleural_task_pred.item(), "baseline": pleural_baseline_pred.item(), "vis": pleural_vis_path, "vis_base64": pleural_vis_base64}}
    print('Finished Validating')
    end_time = time.time()
    duration = end_time - start_time
    print(duration)

    return ret
