from model import EfficientFeebackNetwork
from segformer import SegformerHF_pretrained, SegformerHF_scratch

def getModels(config):
    if config.model==1:
        model = EfficientFeebackNetwork(num_class=config.num_class, in_channels=config.in_channels)
        config.m_name = config.dataset_name + '_EfficientFeebackNetwork'+'_'+str(config.iw)+'x'+str(config.ih)+'_fb-'+str(config.feedback)
        config.checkpoint_path = config.checkpoint_dir + '/' + config.m_name+'.pth'
    elif config.model==2:
        model = SegformerHF_pretrained(num_classes=config.num_class)
        config.m_name = config.dataset_name + '_Segformer_HF_pretrained'+'_'+str(config.iw)+'x'+str(config.ih)+'_fb-'+str(config.feedback)
        config.checkpoint_path = config.checkpoint_dir + '/' + config.m_name+'.pth'
    elif config.model==3:
        model = SegformerHF_scratch(num_classes=config.num_class)
        config.m_name = config.dataset_name + '_Segformer_HF_scratch'+'_'+str(config.iw)+'x'+str(config.ih)+'_fb-'+str(config.feedback)
        config.checkpoint_path = config.checkpoint_dir + '/' + config.m_name+'.pth'
    print(config.m_name)
    return model
