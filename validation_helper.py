import logging
import os.path

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from utils.misc import get_all_preds_labels, get_all_preds_labels_race
import numpy as np
import matplotlib.pyplot as plt
import wandb




def acc_race(dataset, gender, race, label, pred):
    res = {}
    for bio in ['male', 'female']:
        label_name = bio + '_label'
        pred_name = bio + '_pred'
        auc_name = dataset + '_' + bio + '_auc'
        if len(gender[pred_name]) > 0:
            auc = accuracy_score(gender[label_name], np.argmax(gender[pred_name], axis=-1))
        else:
            auc = -1
        res[auc_name] = auc

    for bio in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
        label_name = bio + '_label'
        pred_name = bio + '_pred'
        auc_name = dataset + '_' + bio + '_auc'
        if len(race[pred_name]) > 0:
            auc = accuracy_score(race[label_name], np.argmax(race[pred_name], axis=-1))
        else:
            auc = -1
        res[auc_name] = auc

    intersect_label = gender['intersect_gender_race_labels']
    intersect_preds = gender['intersect_gender_race_preds']
    for gender in ['male', 'female']:
        for race in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            auc_name = dataset + '_' + gender + '_' + race + '_' + 'auc'
            acc_name = dataset + '_' + gender + '_' + race + '_' + 'acc'
            labels = intersect_label[gender][race]
            preds = intersect_preds[gender][race]
            try:
                auc = roc_auc_score(y_true=labels, y_score=preds[:, 1])
            except:
                auc = -1
            if len(preds) > 0:
                acc = accuracy_score(y_true=labels, y_pred=np.argmax(preds, axis=-1))
            else:
                acc = -1
            res[auc_name] = auc
            res[acc_name] = acc

    acc = accuracy_score(label, np.argmax(pred, axis=-1))
    print(dataset, acc, res[dataset + '_male_auc'], res[dataset + '_female_auc'])
    return (dataset, acc, res)


def skew_race(dataset, label, pred, y_gender, y_race):
    pred_label = np.argmax(pred, axis=-1)

    label_real = np.sum(label)
    label_fake = len(label) - label_real

    pred_real = np.sum(pred_label)
    pred_fake = len(pred_label) - pred_real

    result = {
        'real': {
            'male': 0,
            'female': 0,
            'white': 0,
            'black': 0,
            'asian': 0
        },
        'fake': {
            'male': 0,
            'female': 0,
            'white': 0,
            'black': 0,
            'asian': 0
        },
    }

    result_intersect = {
        'real': {
            'male': {
                'white': 0,
                'black': 0,
                'asian': 0
            },
            'female': {
                'white': 0,
                'black': 0,
                'asian': 0
            }
        },
        'fake': {
            'male': {
                'white': 0,
                'black': 0,
                'asian': 0
            },
            'female': {
                'white': 0,
                'black': 0,
                'asian': 0
            }
        }
    }

    for bio in ['male', 'female', 'white', 'black', 'asian']:
        #real
        if bio == 'male' or bio == 'female':
            y = y_gender
        elif bio == 'white' or bio == 'black' or bio == 'asian':
            y = y_race
        label_real_bio = 0
        pred_real_bio = 0
        for ind in range(len(y)):
            if y[ind] == bio and label[ind] == 1:
                label_real_bio += 1
            if y[ind] == bio and pred_label[ind] == 1:
                pred_real_bio += 1
        print(label_real_bio, label_real, pred_real_bio, pred_real)
        p1 = label_real_bio / label_real
        p2 = pred_real_bio / pred_real
        skew_real = np.log(p2 / p1)
        print("Skew of class REAL for bio {bio}: {skew}".format(bio=bio, skew=skew_real))

        #fake
        label_fake_bio = 0
        pred_fake_bio = 0
        for ind in range(len(y)):
            if y[ind] == bio and label[ind] == 0:
                label_fake_bio += 1
            if y[ind] == bio and pred_label[ind] == 0:
                pred_fake_bio += 1
        print(label_fake_bio, label_fake, pred_fake_bio, pred_fake)
        p1 = label_fake_bio / label_fake
        p2 = pred_fake_bio / pred_fake
        skew_fake = np.log(p2 / p1)
        print("Skew of class FAKE for bio {bio}: {skew}".format(bio=bio, skew=skew_fake))

        result['real'][bio] = skew_real
        result['fake'][bio] = skew_fake

    for gender in ['male', 'female']:
        for race in ['white', 'black', 'asian']:
            label_real_bio = 0
            pred_real_bio = 0
            for ind in range(len(y_gender)):
                if y_gender[ind] == gender and y_race[ind] == race:
                    if label[ind] == 1:
                        label_real_bio += 1
                    if pred_label[ind] == 1:
                        pred_real_bio += 1
            print(label_real_bio, label_real, pred_real_bio, pred_real)
            p1 = label_real_bio / label_real
            p2 = pred_real_bio / pred_real
            skew_real = np.log(p2 / p1)
            print("Skew of class REAL for bio {gender}_{race}: {skew}".format(gender=gender, race=race, skew=skew_real))

            label_fake_bio = 0
            pred_fake_bio = 0
            for ind in range(len(y_gender)):
                if y_gender[ind] == gender and y_race[ind] == race:
                    if label[ind] == 0:
                        label_fake_bio += 1
                    if pred_label[ind] == 0:
                        pred_fake_bio += 1
            print(label_fake_bio, label_fake, pred_fake_bio, pred_fake)
            p1 = label_fake_bio / label_fake
            p2 = pred_fake_bio / pred_fake
            skew_fake = np.log(p2 / p1)
            print("Skew of class FAKE for bio {gender}_{race}: {skew}".format(gender=gender, race=race, skew=skew_fake))

            result_intersect['real'][gender][race] = skew_real
            result_intersect['fake'][gender][race] = skew_fake

    values = [value for subdict in result.values() for value in subdict.values()]
    # 计算最大值和最小值
    max_value = max(values)
    min_value = min(values)

    result['maxskew'] = max_value
    result['minskew'] = min_value

    values = [value for subdict in result_intersect['real'].values() for value in subdict.values()]
    # 计算最大值和最小值
    max_value_real = max(values)
    min_value_real = min(values)

    values = [value for subdict in result_intersect['fake'].values() for value in subdict.values()]
    # 计算最大值和最小值
    max_value_fake = max(values)
    min_value_fake = min(values)

    result_intersect['maxskew'] = max(max_value_real, max_value_fake)
    result_intersect['minskew'] = min(min_value_real, min_value_fake)

    print(f"dataset: {dataset}, result: {result}, result_inter: {result_intersect}")

    return [dataset, result, result_intersect]


def validation_race(logger, config, model, test_dataset_list, test_loader_list, device='CUDA', epoch=0):
    test_dataset, test_dfdc_dataset, test_dfd_dataset, test_celeb_dataset = (test_dataset_list['test_dataset'],
                                                                             test_dataset_list['test_dfdc_dataset'],
                                                                             test_dataset_list['test_dfd_dataset'],
                                                                             test_dataset_list['test_celeb_dataset'])

    test_loader, dfdc_loader, dfd_loader, celeb_loader = test_loader_list['test_loader'], test_loader_list[
        'dfdc_loader'], test_loader_list['dfd_loader'], test_loader_list['celeb_loader'],

    logger.info('start the testing...')
    logger.info(f"test_dataset.size: {len(test_dataset)}")
    logger.info(f"dfdc_dataset.size: {len(test_dfdc_dataset)}")
    logger.info(f"df10_dataset.size: {len(test_dfd_dataset)}")
    logger.info(f"celeb_dataset.size: {len(test_celeb_dataset)}")

    y_test, y_test_pred, gender, race, y_test_gender, y_test_race = get_all_preds_labels_race(model, test_loader,
                                                                                              device)
    result = acc_race('ffpp', gender, race, y_test, y_test_pred)
    print(result)

    # print(gender['male_pred'].shape)
    df10_auc, celeb_auc, dfdc_auc, wild_auc = 0.0, 0.0, 0.0, 0.0
    # if not config['data']['split']:
    y_dfdc, y_dfdc_pred, gender_dfdc, race_dfdc, y_dfdc_gender, y_dfdc_race = get_all_preds_labels_race(model,
                                                                                                        dfdc_loader,
                                                                                                        device)
    result_dfdc = acc_race('dfdc', gender_dfdc, race_dfdc, y_dfdc, y_dfdc_pred)
    print(result_dfdc)

    # print(gender_dfdc['male_pred'].shape)
    y_dfd, y_dfd_pred, gender_dfd, race_dfd, y_dfd_gender, y_dfd_race = get_all_preds_labels_race(model, dfd_loader,
                                                                                                  device)
    result_dfd = acc_race('dfd', gender_dfd, race_dfd, y_dfd, y_dfd_pred)
    print(result_dfd)

    y_celeb, y_celeb_pred, gender_celeb, race_celeb, y_celeb_gender, y_celeb_race = get_all_preds_labels_race(model,
                                                                                                              celeb_loader,
                                                                                                              device)
    result_celeb = acc_race('celeb', gender_celeb, race_celeb, y_celeb, y_celeb_pred)
    print(result_celeb)

    skew_result_dfdc = skew_race('dfdc', y_dfdc, y_dfdc_pred, y_dfdc_gender, y_dfdc_race)
    print(skew_result_dfdc)

    skew_result_dfd = skew_race('dfd', y_dfd, y_dfd_pred, y_dfd_gender, y_dfd_race)
    print(skew_result_dfd)

    skew_result_celeb = skew_race('celeb', y_celeb, y_celeb_pred, y_celeb_gender, y_celeb_race)
    print(skew_result_celeb)

    print('*' * 80)
    logger.info('classification report on dfdc set:')
    logger.info(classification_report(y_dfdc, np.argmax(y_dfdc_pred, axis=-1), digits=4))
    logger.info('AUC on dfdc set:')
    dfdc_auc = roc_auc_score(y_true=y_dfdc, y_score=y_dfdc_pred[:, 1])
    dfdc_acc = accuracy_score(y_true=y_dfdc, y_pred=np.argmax(y_dfdc_pred, axis=-1))
    logger.info(dfdc_auc)

    print('*' * 80)
    logger.info('classification report on dfd set:')
    logger.info(classification_report(y_dfd, np.argmax(y_dfd_pred, axis=-1), digits=4))
    logger.info('AUC on df1.0 set:')
    dfd_auc = roc_auc_score(y_true=y_dfd, y_score=y_dfd_pred[:, 1])
    dfd_acc = accuracy_score(y_true=y_dfd, y_pred=np.argmax(y_dfd_pred, axis=-1))
    logger.info(dfd_auc)

    print('*' * 80)
    logger.info('classification report on celeb set:')
    logger.info(classification_report(y_celeb, np.argmax(y_celeb_pred, axis=-1), digits=4))
    logger.info('AUC on celeb set:')
    celeb_auc = roc_auc_score(y_true=y_celeb, y_score=y_celeb_pred[:, 1])
    celeb_acc = accuracy_score(y_true=y_celeb, y_pred=np.argmax(y_celeb_pred, axis=-1))
    logger.info(celeb_auc)

    print('*' * 80)
    logger.info('classification report on ffpp set:')
    logger.info(classification_report(y_test, np.argmax(y_test_pred, axis=-1), digits=4))
    logger.info('AUC on test set:')
    ffpp_acc = accuracy_score(y_true=y_test, y_pred=np.argmax(y_test_pred, axis=-1))
    ffpp_auc = roc_auc_score(y_true=y_test, y_score=y_test_pred[:, 1])
    logger.info(ffpp_auc)

    # print('ACC/AUC for all Dataset: \n FF++: %.4f  %.4f \n DFDC: %.4f  %.4f \n DF1.0: %.4f  %.4f \n Celeb-DF: %.4f  %.4f \n' % (
    #     accuracy_score(y_test, np.argmax(y_test_pred, axis=-1)), ffpp_auc,
    #     accuracy_score(y_dfdc, np.argmax(y_dfdc_pred, axis=-1)), dfdc_auc,
    #     accuracy_score(y_dfd, np.argmax(y_dfd_pred, axis=-1)), dfd_auc,
    #     accuracy_score(y_celeb, np.argmax(y_celeb_pred, axis=-1)), celeb_auc,
    # ))

    auc_list = {'epoch': epoch, 'ffpp_acc': ffpp_acc, 'dfdc_acc': dfdc_acc, 'celeb_acc': celeb_acc, 'dfd_acc': dfd_acc,
                'ffpp_auc': ffpp_auc, 'dfdc_auc': dfdc_auc, 'celeb_auc': celeb_auc, 'dfd_auc': dfd_auc,
                'dfdc_maxskew': skew_result_dfdc[2]['maxskew'], 'dfdc_minskew': skew_result_dfdc[2]['minskew'],
                'dfd_maxskew': skew_result_dfd[2]['maxskew'], 'dfd_minskew': skew_result_dfd[2]['minskew'],
                'celeb_maxskew': skew_result_celeb[2]['maxskew'], 'celeb_minskew': skew_result_celeb[2]['minskew']}

    for gender in ['male', 'female']:
        for race in ['white', 'black', 'asian']:
            auc_name = 'dfdc_{gender}_{race}_auc'.format(gender=gender, race=race)
            auc = result_dfdc[2][auc_name]
            acc_name = 'dfdc_{gender}_{race}_auc'.format(gender=gender, race=race)
            acc = result_dfdc[2][acc_name]
            auc_list[auc_name] = auc
            auc_list[acc_name] = acc

            auc_name = 'dfd_{gender}_{race}_auc'.format(gender=gender, race=race)
            auc = result_dfd[2][auc_name]
            acc_name = 'dfd_{gender}_{race}_auc'.format(gender=gender, race=race)
            acc = result_dfd[2][acc_name]
            auc_list[auc_name] = auc
            auc_list[acc_name] = acc

            auc_name = 'celeb_{gender}_{race}_auc'.format(gender=gender, race=race)
            auc = result_celeb[2][auc_name]
            acc_name = 'celeb_{gender}_{race}_auc'.format(gender=gender, race=race)
            acc = result_celeb[2][acc_name]
            auc_list[auc_name] = auc
            auc_list[acc_name] = acc

    # for gender in ['male', 'female']:
    #     for race in ['black', 'white', 'asian']:
    #         for dataset in ['dfdc', 'dfd', 'celeb']:
    #             key = "skew_{dataset}_{gender}_{race}".format(dataset=dataset, gender=gender, race=race)
    #             auc_list[key] = skew_result[2][cls][gender][race]
    wandb.log(auc_list)

    return auc_list


def save_prediction(pred, label, gender, race):
    all_pred_low = pred  # list or numpy array (N,)
    all_label = label  # (N,)
    all_gender = gender  # (N,)
    all_race = race  # (N,)

    # 保存到磁盘，供下一次模型读取
    np.savez("./cache_high.npz",
             pred_high=np.asarray(all_pred_low, dtype=np.float32),
             label=np.asarray(all_label, dtype=np.int8),
             gender=np.asarray(all_gender),
             race=np.asarray(all_race))


def add_data(dataset, y, y_pred, y_gender, y_race, table_skew):
    # add_skew_result
    skew_result = skew_race(dataset, y, y_pred, y_gender, y_race)
    table_skew.add_data(
        skew_result[0], skew_result[1]['maxskew'], skew_result[1]['minskew'], 'real',
        skew_result[1]['real']['male'], skew_result[1]['real']['female'],
        skew_result[1]['real']['white'], skew_result[1]['real']['black'],
        skew_result[1]['real']['asian'],
    )
    table_skew.add_data(
        skew_result[0], skew_result[1]['maxskew'], skew_result[1]['minskew'], 'fake',
        skew_result[1]['fake']['male'], skew_result[1]['fake']['female'],
        skew_result[1]['fake']['white'], skew_result[1]['fake']['black'],
        skew_result[1]['fake']['asian']
    )
    return skew_result


def write_score(submitted_score_file, all_pred, all_path, all_ori_path):
    dict_score = {}

    # with open(submitted_score_file) as rd:
    #     lines = rd.readlines()
    #     for line in lines:
    #         print(line)
    #         path, score = line.split('	')
    #         full_path = os.path.join('/home/harryxa', path)
    #         dict_score[full_path] = path

    with open('./submitted_scores.txt', 'w') as wr:
        for ind in range(len(all_ori_path)):
            path_real = os.path.join('/', all_ori_path[ind][1:])
            pred_real = all_pred[ind][1]

            wr.write('%s\t%f\n' % (path_real, pred_real))


def test_race(logger, config, model, test_dataset_list, test_loader_list, device='CUDA', epoch=None):
    test_dataset = (test_dataset_list['test_dataset'])
    test_loader = test_loader_list['test_loader']

    table = wandb.Table(columns=["Dataset", "Acc"])
    table_skew = wandb.Table(columns=["Dataset", "max_skew", "min_skew", "label"])

    logger.info('start the testing...')
    logger.info(f"test_dataset.size: {len(test_dataset)}")

    y_test, y_test_pred, gender, race, path, ori_path = get_all_preds_labels_race(model, test_loader, device)
    # result_ffpp = acc_race('AIFace', gender, race, y_test, y_test_pred)

    # add_data('AIFace', y_test, y_test_pred, y_test_gender, y_test_race, table_skew)

    # wandb.log({
    #     'table': table,
    #     'table_skew': table_skew
    # })

    print('*' * 80)
    logger.info('classification report on AIFace set:')
    logger.info(classification_report(y_test, np.argmax(y_test_pred, axis=-1), digits=4))
    logger.info('AUC on test set:')
    ffpp_acc = accuracy_score(y_true=y_test, y_pred=np.argmax(y_test_pred, axis=-1))
    ffpp_auc = roc_auc_score(y_true=y_test, y_score=y_test_pred[:, 1])
    logger.info(ffpp_auc)

    write_score('/home/harryxa/AI_Face_imagesV2/submitted_scores.txt', all_path=path, all_pred=y_test_pred,
                all_ori_path=ori_path)

    auc_list = {'epoch': epoch, 'acc': ffpp_acc, 'auc': ffpp_auc}

    # for gender in ['male', 'female']:
    #     for race in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
    #         auc_name = 'ffpp_{gender}_{race}_auc'.format(gender=gender, race=race)
    #         auc = result_ffpp[2][auc_name]
    #         acc_name = 'ffpp_{gender}_{race}_auc'.format(gender=gender, race=race)
    #         acc = result_ffpp[2][acc_name]
    #         auc_list[auc_name] = auc
    #         auc_list[acc_name] = acc

    wandb.log(auc_list)

    return auc_list
