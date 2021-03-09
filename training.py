import torch
from torch.utils.data import DataLoader, ConcatDataset

from configs.train_config import SpectConfig, TrainConfig
from configs.test_config import TestConfig
from models.model import Conformer
from loader.data_loader import SpectrogramDataset, collate_fn


def train(train_loader, num_epochs, start_epoch=1, ):
    
    for epoch in range(start_epoch, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_predict = []
        valid_predict = []
        train_target = []
        valid_target = []
        temp_predict = []
        temp_target = []
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, batch_data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            # move to GPU
            data, attn_mask, lenghts = batch_data[0].to(device=device), batch_data[1].to(device=device), batch_data[2].to(device=device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data, attn_mask, lengths)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            ## record the average training loss, using something like
            _, predictions = output.max(1)
            temp_predict = [pred.item() for pred in predictions]
            temp_target = [actual.item() for actual in target]

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            train_predict = train_predict + temp_predict
            train_target = train_target + temp_target

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, batch_data in tqdm(enumerate(test_loader), total=len(loaders['test']), leave=False):
            # move to GPU
 
            data, attn_mask, lengths = batch_data[0].to(device=device), batch_data[1].to(device=device), batch_data[2].to(device=device)
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data, attn_mask, lengths)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            _, predictions = output.max(1)
            temp_predict = [pred.item() for pred in predictions]
            temp_target = [actual.item() for actual in target]

            valid_predict = valid_predict + temp_predict
            valid_target = valid_target + temp_target

        # calculate average losses
        train_loss = train_loss / len(loaders['train'].dataset)
        valid_loss = valid_loss / len(loaders['test'].dataset)
        train_acc = accuracy_score(train_target, train_predict)
        valid_acc = accuracy_score(valid_target, valid_predict)

        # print training/validation statistics
        print(
            'Epoch: {} \tTraining Loss: {:.10f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.10f} \tValidation  Accuracy: {:.6f} '.format(
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc
            ))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        scheduler.step(valid_loss)
        # save checkpoint
        save_ckp(checkpoint, model, valid_loss, valid_loss_min, checkpoint_path, best_model_path, final_model_path,
                 save_for_each_epoch)

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).    Model Saved......'.format(valid_loss_min, valid_loss))
            # save_ckp(checkpoint, model, True, checkpoint_path, best_model_path, final_model_path)
            valid_loss_min = valid_loss
    return model

def validation():
    pass

if __name__ == "__main__":

    spect_config = SpectConfig()

    # The train dataset
    train_config = TrainConfig()
    speech_train_path = train_config.speech_folder_path
    speech_train_dataset = SpectrogramDataset(speech_train_path, 0, spect_config)
    songs_train_path = train_config.speech_folder_path
    songs_train_datset = SpectrogramDataset(songs_train_path, 1, spect_config)
    train_dataset = ConcatDataset[speech_train_dataset, songs_train_datset]

    # The test dataset
    test_config = TestConfig() 
    speech_test_path = test_config.speech_folder_path
    speech_test_dataset = SpectrogramDataset(speech_test_path, 0, spect_config)
    songs_test_path = test_config.songs_folder_path
    songs_test_dataset = SpectrogramDataset(songs_test_path, 1, spect_config)
    test_dataset = ConcatDataset([speech_test_dataset, songs_test_dataset])

    # The train and test dataloaders
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, collate_fn=collate_fn,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=test_config.batch_size, collate_fn=collate_fn,
                             pin_memory=True)

    train(train_loader, train_confg.num_epochs)