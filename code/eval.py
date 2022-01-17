import os
import json
from glob import glob

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import timm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
import albumentations as A
from albumentations.pytorch import ToTensorV2


ROOT_DIR = '../data'

def initialize():
    csv_feature_dict = {
        '내부 온도 1 평균': [3.4, 47.3],
        '내부 온도 1 최고': [3.4, 47.6],
        '내부 온도 1 최저': [3.3, 47.0],
        '내부 습도 1 평균': [23.7, 100.0],
        '내부 습도 1 최고': [25.9, 100.0],
        '내부 습도 1 최저': [0.0, 100.0],
        '내부 이슬점 평균': [0.1, 34.5],
        '내부 이슬점 최고': [0.2, 34.7],
        '내부 이슬점 최저': [0.0, 34.4]
    }
    
    crop = {'1': '딸기', '2': '토마토', '3': '파프리카', '4': '오이', '5': '고추', '6': '시설포도'}
    disease = {
        '1': {
            'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해', 
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '2': {
            'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍',
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '3': {
            'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍', 
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '4': {
            'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해', 
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)' 
        },
        '5': {
            'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍', 
            'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
        },
        '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}
    }
    risk = {'1': '초기', '2': '중기', '3': '말기'}
    
    label_description = {}
    for key, value in disease.items():
        label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
        for disease_code in value:
            for risk_code in risk:
                label = f'{key}_{disease_code}_{risk_code}'
                label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'
                
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_decoder = {val: key for key, val in label_encoder.items()}
    
    return csv_feature_dict, label_encoder, label_decoder


def split_data(split_rate=0.2, seed=42, mode='train'):
    """
    Use for model trained image and time series.
    """
    if mode == 'train':
        train = sorted(glob(f'{ROOT_DIR}/train/*'))
        
        labelsss = pd.read_csv(f'{ROOT_DIR}/train.csv')['label']
        train, val = train_test_split(
            train, test_size=split_rate, random_state=seed, stratify=labelsss)
        
        return train, val
    elif mode == 'test':
        test = sorted(glob(f'{ROOT_DIR}/test/*'))

        return test
    
    
csv_feature_dict, label_encoder, label_decoder = initialize()

SEED = 42
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
BATCH_SIZE = 8
CLASS_N = len(label_encoder)
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 512
NUM_FEATURES = len(csv_feature_dict)
MAX_LEN = 24*6
DROPOUT_RATE = 0.1
EPOCHS = 10
NUM_WORKERS = 2

torch.multiprocessing.set_sharing_strategy('file_system')


class CustomDataset(Dataset):
    def __init__(
        self, 
        files, 
        csv_feature_dict, 
        label_encoder,
        labels=None,
        transforms=None,
        mode='train',
    ):
        self.mode = mode
        self.files = files
        
        self.csv_feature_dict = csv_feature_dict
        
        if files is not None:
            self.csv_feature_check = [0]*len(self.files)
            self.csv_features = [None]*len(self.files)
            
        self.max_len = 24 * 6
        
        self.label_encoder = label_encoder
        
        self.transforms = transforms

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split(os.sep)[-1]
        
        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]
        
        # image
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        if self.mode == 'train':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'
            
            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32),
                'label': torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return {
                'img': img,
                'csv_feature': torch.tensor(csv_feature, dtype=torch.float32)
            }
    
    
class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        train=None,
        val=None,
        test=None,
        csv_feature_dict=None,
        label_encoder=None,
        train_transforms=None,
        val_transforms=None,
        predict_transforms=None,
        num_workers=32,
        batch_size=8,
    ):
        super().__init__()
        self.train = train
        self.val = val
        self.test = test
        self.csv_feature_dict = csv_feature_dict
        self.label_encoder = label_encoder
        assert self.csv_feature_dict is not None
        assert self.label_encoder is not None
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.predict_transforms = predict_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(
            self.train, 
            self.csv_feature_dict,
            self.label_encoder,
            transforms=self.train_transforms,
        )
        self.valid_dataset = CustomDataset(
            self.val, 
            self.csv_feature_dict,
            self.label_encoder,
            transforms=self.train_transforms,
        )
        self.predict_dataset = CustomDataset(
            self.test, 
            self.csv_feature_dict,
            self.label_encoder,
            transforms=self.predict_transforms,
            mode='test'
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
        
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )

def accuracy_function(real, pred):    
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score


class CNN_Encoder(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN_Encoder, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1000)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output
    
    
class LSTM_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(LSTM_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, 1000)
        self.final_layer = nn.Linear(1000 + 1000, class_n)  # resnet out_dim + lstm out_dim
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1)  # enc_out + hidden 
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output
    
    
class BaseModel(LightningModule):
    def __init__(
        self,
        cnn,
        rnn,
        criterion,
        learning_rate=5e-4,
    ):
        super(BaseModel, self).__init__()
        
        self.cnn = cnn
        self.rnn = rnn
        self.learning_rate = learning_rate
        self.criterion = criterion
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            max_lr=lr,
            epochs=max_epochs,
            optimizer=optimizer,
            steps_per_epoch=int(len(train_dataset) / batch_size),
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
            base_momentum=0.90,
            max_momentum=0.95,
        )
        return [optimizer], [scheduler]

    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)
        
        return output

    def training_step(self, batch, batch_idx):
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']
        
        output = self(img, csv_feature)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)
        
        self.log(
            'train_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'train_score', score, prog_bar=True, logger=True
        )
        
        return {'loss': loss, 'train_score': score}        

    def validation_step(self, batch, batch_idx):
        img = batch['img']
        csv_feature = batch['csv_feature']
        label = batch['label']
        
        output = self(img, csv_feature)
        loss = self.criterion(output, label)
        score = accuracy_function(label, output)
        
        self.log(
            'val_loss', loss, prog_bar=True, logger=True
        )
        self.log(
            'val_score', score, prog_bar=True, logger=True
        )
        
        return {'val_loss': loss, 'val_score': score}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img = batch['img']
        seq = batch['csv_feature']
        
        output = self(img, seq)
        output = torch.argmax(output, dim=1)
        
        return output


class CNN2RNNModel(BaseModel):
    def __init__(
        self,
        max_len, 
        embedding_dim, 
        num_features, 
        class_n,
        rate=0.1,
        learning_rate=5e-4,
    ):
        cnn = CNN_Encoder(class_n)
        rnn = LSTM_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
        criterion = nn.CrossEntropyLoss()
        
        super(CNN2RNNModel, self).__init__(
            cnn, rnn, criterion, learning_rate
        )

def get_train_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_valid_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def train(model_name, csv_feature_dict, label_encoder, seed=42):
    """
    Use for model trained image and time series.
    """
    train_data, val_data = split_data(seed=seed, mode='train')
    
    train_transforms = get_train_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)
    val_transforms = get_valid_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    data_module = CustomDataModule(
        train=train_data,
        val=val_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
    )
    
    model = CNN2RNNModel(
        max_len=MAX_LEN, 
        embedding_dim=EMBEDDING_DIM, 
        num_features=NUM_FEATURES, 
        class_n=CLASS_N, 
        rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )
    
    ckpt_path = f'./weights/{model_name}/'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    checkpoint = ModelCheckpoint(
        monitor='val_score',
        dirpath=ckpt_path,
        filename='{epoch}-{val_score:.3f}',
        save_top_k=-1,
        mode='max',
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gpus=1,
        precision=16,
        callbacks=[checkpoint],
        log_every_n_steps=5,
    )

    trainer.fit(model, data_module)

def get_predict_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_submission(outputs, save_dir, save_filename, label_decoder):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    outputs = [o.detach().cpu().numpy() for batch in outputs
                                        for o in batch]
    preds = np.array([label_decoder[int(val)] for val in outputs])
    
    submission = pd.read_csv(f'{ROOT_DIR}/sample_submission.csv')
    submission['label'] = preds
    
    save_file_path = os.path.join(save_dir, save_filename)
    
    submission.to_csv(save_file_path, index=False)


def eval(
    ckpt_path, 
    csv_feature_dict, 
    label_encoder, 
    label_decoder,
    submit_save_dir='submissions',
    submit_save_name='baseline_submission.csv',
):
    test_data = split_data(mode='test')
    
    predict_transforms = get_predict_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)
    
    data_module = CustomDataModule(
        test=test_data,
        csv_feature_dict=csv_feature_dict,
        label_encoder=label_encoder,
        predict_transforms=predict_transforms,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
    )
    
    model = CNN2RNNModel(
        max_len=24*6, 
        embedding_dim=512, 
        num_features=len(csv_feature_dict), 
        class_n=len(label_encoder),
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
    )

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])

    outputs = trainer.predict(model, data_module)

    get_submission(outputs, submit_save_dir, submit_save_name, label_decoder)

    seed_everything(SEED)

MODEL_NAME = 'eff'

#train(MODEL_NAME, csv_feature_dict, label_encoder, seed=SEED)

CKPT_PATH = 'weights/eff/epoch=9-val_score=0.935.ckpt'

eval(CKPT_PATH, csv_feature_dict, label_encoder, label_decoder)    