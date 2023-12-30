from configs import Configs
from kvasir_loader import KVASIR_loader
from cvc_clinicdb_loader import CVC_ClinicDB_loader
from bus_loader import BUS_loader
from isic_loader import ISIC_loader
from cvc300_loader import CVC300_loader
from colondb_loader import ColonDB_loader
from get_models import getModels
#from get_models_no_feedback import getModels
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from trainer import Trainer

if __name__=='__main__':
    print(1)
    config = Configs().parse()

    config.dataset_name = 'OUT_DOMAIN'
    config.in_channels=3
    train_data1 = KVASIR_loader(augmentation=True)
    val_data1 = KVASIR_loader(phase='val')
    
    train_data2 = CVC_ClinicDB_loader(augmentation=True,iw=512,ih=512)
    val_data2 = CVC_ClinicDB_loader(phase='val',iw=512,ih=512)

    train_data = torch.utils.data.ConcatDataset([train_data1,train_data2])
    val_data = torch.utils.data.ConcatDataset([val_data1,val_data2])

    if config.train:
        config.in_channels=3
        test_data1 = ColonDB_loader()
        test_data2 = CVC300_loader()
        test_data = torch.utils.data.ConcatDataset([test_data1,test_data2])
    else:
        if config.out_domain=='CVC-ColonDB':
            config.in_channels=3
            test_data = ColonDB_loader()
        elif config.out_domain=='CVC300':
            config.in_channels=3
            test_data = CVC300_loader()

    train_loader = DataLoader(train_data,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,drop_last=True)
    val_loader = DataLoader(val_data,batch_size=config.BATCH_SIZE,num_workers=config.NUM_WORKERS,shuffle=False,drop_last=True)
    test_loader = DataLoader(test_data,batch_size=config.BATCH_SIZE,num_workers=config.NUM_WORKERS,shuffle=False,drop_last=True)

    model = getModels(config=config)
    if config.parallel:
        model = torch.nn.DataParallel(model,device_ids=[0,1])
    model.cuda()

 
    if config.num_class==2:
        weight = torch.ones(config.num_class).to(config.device)
        weight[0] = 0.3
        weight[1] = 0.7
    else:
        weight = torch.ones(config.num_class).to(config.device)
        weight[0] = 0.3
        weight[1:] = 0.7

    config.weight = weight

    if config.wloss:
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        config=config,
        criterion=criterion,
        optimizer=optim.AdamW(model.parameters(),lr=config.lr),
    )

    if config.train:
        trainer.fit(train_loader=train_loader,val_loader=val_loader)
    
    if config.test:
        trainer.evaluator_out_domain(test_loader=test_loader)