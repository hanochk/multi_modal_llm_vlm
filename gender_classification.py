import os
import glob
# import pandas as pd
from PIL import Image
import numpy as np
import clip #pip install git+https://github.com/openai/CLIP.git
import tqdm
import sklearn.metrics
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset as Dataset
import torch.nn as nn
import torch.optim as optim

def to_device(tensor_or_list, device):
    if isinstance(tensor_or_list, (list, tuple)):
        tensor_or_list = [tensor.to(device) for tensor in tensor_or_list]
    else:
        tensor_or_list = tensor_or_list.to(device)

    return tensor_or_list

class Data(Dataset):
    def __init__(self, img_preprocess, tokenizer, image_path: list[str], text_list: list[str], **kwargs): # add super for father class
        super(Dataset, self).__init__()

        self.img_preprocess = img_preprocess
        self.image_path = image_path
        self.text_tokens  = tokenizer(["A photo of a {}".format(x) for x in text_list]) #        ]
        self.class_text = text_list
        self.is_classifier_uniq_cls = kwargs.pop('classifier_uniq_cls', False)
        if self.is_classifier_uniq_cls:
            self.classes = np.unique(text_list)
            self.classifier_uniq_cls_tokens = tokenizer(["A photo of a {}".format(x) for x in self.classes])

        assert(len(self.image_path) == len(self.text_tokens))
        return

    def __len__(self):
        return len(self.text_tokens)   

    def __getitem__(self, idx: int):
        if self.is_classifier_uniq_cls:
            image = self.img_preprocess(Image.open(self.image_path[idx]))
            label = np.where(self.classes == self.class_text[idx])[0].item()
            return image, label, torch.tensor([-1])
        else:
            image = self.img_preprocess(Image.open(self.image_path[idx]))
            text = self.text_tokens[idx]
            label = [0 if self.class_text[idx] == 'female' else 1] # 0 -feamle 1- male no offence :-)
            return image, label, text

# pip install git+https://github.com/openai/CLIP.git
class ClipVideoUtils:
    def __init__(self, device, batch_size:int =1024, model_name:str ='RN50x4'): 
        """
        The optimal batch size depends on the GPU. If CPU, it's zero
        """
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.batch_size = batch_size if torch.cuda.is_available() else 0
        self._convert_weights_back_to_fp16 = clip.model.convert_weights
    
    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, a):
        self._train_dataset = a

    @property
    def val_dataset(self):
        return self._val_dataset

    @train_dataset.setter
    def val_dataset(self, a):
        self._val_dataset = a

    def train_model_contrastive(self, dataloader: Data, loss_img, optimizer, num_epochs: int=1, **kwargs):
        
        #https://github.com/openai/CLIP/issues/57
        def convert_models_to_fp32(model): 
            for p in model.parameters(): 
                p.data = p.data.float() 
                if p.grad != None:
                    p.grad.data = p.grad.data.float() 

        max_iter = kwargs.pop('max_iterations', -1)
        all_targets = list()
        all_simillarity = list()
        all_predictions = list()
        epoch_loss = list()
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            # TODO eval process after each epoch
            for iters, (images, labels, texts) in enumerate(tqdm.tqdm(dataloader)):

                if (max_iter != -1 and iters > max_iter):
                    break
                optimizer.zero_grad()                
                image_input = to_device(images, self.device)
    # text_inputs = to_device(clip.tokenize(["A photo of a {}".format(x) for x in text]), self.device)
                if dataloader.dataset.is_classifier_uniq_cls:
                    text_features = self.model.encode_text(to_device(dataloader.dataset.classifier_uniq_cls_tokens, self.device))             # TODO : avoid re-encode pre-defined texts just read from local tensor
                else:
                    text_inputs = to_device(texts, self.device)
                    text_features = self.model.encode_text(text_inputs)             # TODO : avoid re-encode pre-defined texts just read from local tensor
# 
                # cosine similarity as logits forward method()   #https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#LL366C9-L366C38
                image_features = self.model.encode_image(image_input)
                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = 1 #self.model.logit_scale.exp() # temperature should stay low for bin classifier
                logits_per_image = logit_scale * image_features @ text_features.t()
                # logits_per_text = logits_per_image.t()
                
                ground_truth = to_device(labels[0], self.device).long() #torch.arange(len(images),dtype=torch.long, device=self.device)

                total_loss = loss_img(logits_per_image, ground_truth)
                total_loss.backward()
                if self.device == "cpu":
                    optimizer.step()
                else: 
                    convert_models_to_fp32(self.model)
                    optimizer.step()
                    self._convert_weights_back_to_fp16(self.model)
# 
        # Hidden assumption: The loss has a "mean" reduction (see pytorch loss arguments)
                running_loss += total_loss.item() * dataloader.batch_size
            epoch_loss.append(running_loss / len(dataloader.dataset))
        return epoch_loss ,running_loss

    def eval_model_contrastive(self, dataloader: Data, **kwargs):
        all_targets = list()
        all_simillarity = list()
        all_predictions = list()
        
        self.model.eval()

        with torch.no_grad():
            for images, targets, texts in tqdm.tqdm(dataloader):
                image_input = to_device(images, self.device)
    # text_inputs = to_device(clip.tokenize(["A photo of a {}".format(x) for x in text]), self.device)
                if dataloader.dataset.is_classifier_uniq_cls:
                    text_features = self.model.encode_text(to_device(dataloader.dataset.classifier_uniq_cls_tokens, self.device))             # TBD : avoid re-encode pre-defined texts just read from local tensor
                else:
                    text_inputs = to_device(texts, self.device)
                    text_features = self.model.encode_text(text_inputs)             # TBD : avoid re-encode pre-defined texts just read from local tensor

                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True) # normalization is not mandatory for argmax()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T)
                max_sim_ind = np.argmax(similarity.cpu().numpy(), axis=1)

                all_simillarity.append(similarity.cpu().numpy())
                all_predictions.append(max_sim_ind)
                all_targets.append(targets)

            all_targets = np.concatenate(all_targets)
            all_predictions = np.concatenate(all_predictions)
            all_simillarity = np.concatenate(all_simillarity)


            return all_targets, all_predictions, all_simillarity


    # def eval_model_cusstome(self, loss_fn=None, **kwargs):
    #     self.model.eval()

    #     with torch.no_grad():
    #         for read_list in tqdm.tqdm(dataloader):
    #             # TODO: NORMALIZATION! In case we add something else than hardcoded normalization
    #             # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
    #             #                                          shuffle=False)
    #             # Check if model in cuda : next(model.parameters()).is_cuda  => should be True
    #             total_val_loss = 0.0
    #             n_samples = 0

    #             all_targets = list()
    #             all_predictions = list()
    #             all_features = list()
    #             batch_counter = 0
    #             all_tile_id = list()
    #             all_atten_weithgs = list()
    #             all_tile_index_pos = list()

    #             for read_list in tqdm.tqdm(dataloader):
    #                 inputs, targets = read_list
    #                 inputs = to_device(inputs, device)

    #                 if kwargs['extract_features']:
    #                     if hasattr(model, '_custom_forward_impl'): # model that can combines Hand HCF should have this method implemented to combine HCF
    #                         x = model.features(inputs) # running model double time, didn;t find a way to resue features
    #                         features = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)  # copied rfom mobilenet.py
    #                         predictions = model.forward(inputs)
    #                     else:
    #                         x = model.features(inputs[0])
    #                         features = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1) # copied rfom mobilenet.py
    #                         predictions = model.forward(inputs[0])

    #                     # predictions, features = model.forward_return_multiple_features(inputs)
    #                 else:
    #                     if hasattr(model, '_custom_forward_impl') or hasattr(model, '_custom_forward_impl_embeddings_pooling'): # model that can combines Hand HCF should have this method implemented to combine HCF
    #                         predictions = model.forward(inputs)
    #                     else:
    #                         predictions = model.forward(inputs[0])

    #                     features = None

    #                 if loss_fn is not None:
    #                     val_loss = loss_fn(predictions, targets)
    #                     n_samples_in_batch = len(targets)
    #                     n_samples += n_samples_in_batch
    #                     # Hidden assumption: The loss has a "mean" reduction (see pytorch loss arguments)
    #                     total_val_loss += val_loss.item() * n_samples_in_batch

    #                 if do_softmax:
    #                     predictions = torch.nn.functional.softmax(predictions, dim=1)
    #                     # print(predictions[:,1])
    #                 else:
    #                     predictions = predictions # logits

    #             all_targets.append(targets.cpu().numpy())
    #             all_predictions.append(predictions.cpu().numpy())


    #             batch_counter += 1
    #             if batch_counter == max_number_of_batches:
    #                 break
    #         all_targets = np.concatenate(all_targets)
    #         all_predictions = np.concatenate(all_predictions)
    #     loss = total_val_loss / n_samples if loss_fn is not None else None


    
def data_factory(img_preprocess, val_image_path: str, train_image_path: str, batch_size: int, num_workers: int):

    train_image_fname = glob.glob(train_image_path + '/**/*.jpg', recursive=True) + glob.glob(train_image_path + '/**/*.png', recursive=True) + \
                    glob.glob(train_image_path + '/**/*.jpeg', recursive=True)

    val_image_fname = glob.glob(val_image_path + '/**/*.jpg', recursive=True) + glob.glob(val_image_path + '/**/*.png', recursive=True) + \
                    glob.glob(val_image_path + '/**/*.jpeg', recursive=True)

    train_image_text = [x.split('/')[-2] for x in train_image_fname]

    val_image_text = [x.split('/')[-2] for x in val_image_fname]

    train_dataset = Data(img_preprocess=img_preprocess, tokenizer=clip.tokenize, 
                                image_path=train_image_fname, text_list=train_image_text, classifier_uniq_cls=False) # take the 
    
    val_dataset = Data(img_preprocess=img_preprocess, tokenizer=clip.tokenize, 
                                image_path=val_image_fname, text_list=val_image_text, classifier_uniq_cls=True)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)
    
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers)


    return train_dataloader, val_dataloader

def main():
    # Params to be moved to argsparse/user_defined 
    batch_size = int(128/8) # to be optimized with respect to GPU
    num_workers = 8
    result_dir = '/notebooks/multi_modal'
    state_file_name = 'clip_gender_ft_cls.pt'
    eval_at_1st = False
    data_path = '/notebooks/gender_dataset'
    # data_path = '/notebooks/gender_dataset/tiny_set'

    device = "cuda" if torch.cuda.is_available() else "cpu"
# Data handler : Based on this data set https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
    clip_obj = ClipVideoUtils(device=device, batch_size=batch_size)

    val_image_path = os.path.join(data_path, 'Validation')
    train_image_path = os.path.join(data_path, 'Training')
    
    train_dataloader, val_dataloader = data_factory(img_preprocess=clip_obj.preprocess, 
                                                    val_image_path=val_image_path, 
                                                    train_image_path=train_image_path,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers)

    if eval_at_1st:
        all_targets, all_predictions, all_simillarity = clip_obj.eval_model_contrastive(dataloader=val_dataloader)
        acc =  np.where(all_targets-all_predictions == 0)[0].shape[0]/all_targets.shape[0]# TODO consider replace with metrics.auc thresholding over the similarity score !!!
        print("ACC: {} over validation set {} images".format(acc, all_targets.shape[0]))
        if not np.isclose(acc, 0.9672933298995622):
            raise ValueError("something wrong with model or data has changed ")
        # FT model wit hmoderate Lr
    loss_img = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clip_obj.model.parameters(), lr=1e-5, weight_decay=0.001) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    running_loss = clip_obj.train_model_contrastive(dataloader=train_dataloader,optimizer=optimizer, 
                                                    loss_img=loss_img, max_iterations=1000)

    torch.save({
        'epoch': -1,
        'model_state_dict': clip_obj.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
        },os.path.join(result_dir, state_file_name))

    all_targets, all_predictions, all_simillarity = clip_obj.eval_model_contrastive(dataloader=val_dataloader)
    acc =  np.where(all_targets-all_predictions == 0)[0].shape[0]/all_targets.shape[0]# TODO consider replace with metrics.auc thresholding over the similarity score !!!
    print("ACC: {} after FT over validation set {} images".format(acc, all_targets.shape[0]))


    return


if __name__ == '__main__':
    main()

# RN50x4
# RN50x4
# https://github.com/openai/CLIP/issues/83

# RN50x4 model expects an 288x288 image input