import os
from dataset.ImageFolder import ImageFolder_race
from torch.utils.data import DataLoader



def data_load_race(logger, config, tranforms):
    train_dataset = ImageFolder_race(os.path.join(config['data']['data_path_train'], "train"),
                                     csv_list=config['data']['csv_list_ffpp']['train'],
                                     config=config,
                                     mask_path=config['data']['mask_path'],
                                     transform=tranforms['train'],
                                     bio_choose=config['data']['bio_choose'],
                                     phase='train'
                                     )

    val_dataset = ImageFolder_race(os.path.join(config['data']['data_path_val'], "test"),
                                   csv_list=config['data']['csv_list_ffpp']['val'],
                                   config=config,
                                   mask_path=config['data']['mask_path'],
                                   transform=tranforms['val'],
                                   phase='test')

    test_dataset = ImageFolder_race(os.path.join(config['data']['data_path_test'], "test"),
                                    csv_list=config['data']['csv_list_ffpp']['val'],
                                    config=config,
                                    mask_path=config['data']['mask_path'],
                                    transform=tranforms['test'],
                                    phase='test')


    logger.info(f"train_dataset.classes: {train_dataset.classes}")
    logger.info(f"train_dataset.size: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=2)

    train_dataset_list = {
        'train_dataset': train_dataset
    }

    val_dataset_list = {
        'val_dataset': val_dataset
    }

    test_dataset_list = {
        'test_dataset': test_dataset,
    }

    train_loader_list = {
        'train_loader': train_loader
    }

    val_loader_list = {
        'val_loader': val_loader,
    }

    test_loader_list = {
        'test_loader': test_loader,
    }

    return train_dataset_list, val_dataset_list, test_dataset_list, train_loader_list, val_loader_list, test_loader_list
