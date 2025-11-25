
def train_one_epoch(model, loss_fn, optimizer, dataloader, device):
    """
    Train a pytorch model for one epoch
    :param model:
    :param loss_fn:
    :param optimizer:
    :param dataloader:
    :param device:
    :return:
    """

    for imgs, labels in dataloader:
        # bring image to gpu if available
        imgs.to(device)

        outputs = model(imgs)

        optimizer.zero_grad()

        # compute loss and do backward propagation
        loss = loss_fn(outputs, labels)
        loss.backward()

        # adjust model weights
        optimizer.step()
