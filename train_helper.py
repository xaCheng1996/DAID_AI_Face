import time
import torch
from tqdm import tqdm
from utils.misc import AverageMeter, set_seed, save_checkpoint, get_all_preds_labels
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score




def train_effi(config, model, optimizer, criterion, criterion_cos, train_loader, device=torch.device('cpu'), scaler=None, lambda_ortho=0.2):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_mask = AverageMeter()
    losses_contras = AverageMeter()
    end = time.time()
    batch_size = 0
    model.train()

    for (inputs, targets, _, _, inputs_contras, weight) in tqdm(train_loader):
        inputs, targets, weight = inputs.to(device), targets.to(device), weight.to(device)
        inputs_contras = inputs_contras.to(device)

        data_time.update(time.time() - end)

        if lambda_ortho:
            # print('using orthogonal loss')
            logits, features = model(inputs)
            if config['data']['reduction'] == 'none':
                cls_loss = criterion(logits, targets)
                loss_weighted = cls_loss * weight
                cls_loss = loss_weighted.mean()
            else:
                cls_loss = criterion(logits, targets)

            svd_loss = model.singular_value_regularization(mode='sparse', r=128) + model.singular_value_regularization(mode='tail', r=128)
            ortho_loss = model.orthogonality_regularization()
            loss_ori = cls_loss + lambda_ortho * ortho_loss + lambda_ortho * svd_loss

            logits_contras, features_contras = model(inputs_contras)
            target_contras = torch.ones(features.size(0)).to(features.device)
            loss_contras = criterion_cos(features, features_contras, target_contras) + 1e-6
            if config['data']['reduction'] == 'none':
                loss_cls_contras = criterion(logits_contras, targets)
                loss_weighted = loss_cls_contras
                loss_cls_contras = loss_weighted.mean()
            else:
                loss_cls_contras = criterion(logits_contras, targets)
            ortho_loss = model.orthogonality_regularization()
            loss_cls_contras = loss_cls_contras + lambda_ortho * ortho_loss + lambda_ortho * svd_loss

            final_loss = loss_ori + loss_cls_contras + 0.7 * loss_contras

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            losses_contras.update(loss_contras.detach().cpu().numpy(), inputs.size(0))
        # return loss.item(), cls_loss.item(), ortho_loss.item()

        else:
            # with autocast('cuda'):
            logits = model(inputs)
            # features = model.extract_features(inputs)
            # print(logits)
            # print(targets, weight)
            loss = criterion(logits, targets)
            final_loss = loss
            # loss__weighted = loss * weight
            # final_loss = loss__weighted.mean()
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            # scaler.scale(loss).backward(retain_graph=True)
            # scaler.step(optimizer)
            # scaler.update()
            final_loss.backward(retain_graph=True)
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(final_loss.item(), inputs.size(0))
        losses.update(final_loss.item())
        # if batch_idx % 100 == 0:
        #     logger.info(f"batch idx: {batch_idx}, train loss: {loss}")
    print("batch time: %.4f, data_time: %.4f" % (batch_time.avg, data_time.avg))
    print("video throughout during training: %.4f videos/s" % (train_loader.batch_size / batch_time.avg))
    return losses.avg, losses_mask.avg


