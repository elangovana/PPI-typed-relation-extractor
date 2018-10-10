import torch
import torch.utils.data

class Train:

    def __init__(self):
        pass

    def __call__(self, data_iter, model_network, loss_function, optimizer, epoch=100, mini_batch_size=1):
        """
Runs train...
        :param data_iter: An iterable record of tuples of the fomw (label, features).  The each feature must be the index of word vocab
        :param model_network: A neural network
        :param loss_function: Pytorch loss function
        :param optimizer: Optimiser
        """
        losses = []
        train_loader = torch.utils.data.DataLoader(data_iter, batch_size=mini_batch_size, shuffle=True, )

        for epoch in range(epoch):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                #for feature, target in zip(batch_x, batch_y):
                    # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                    # into integer indices and wrap them in tensors)
                    context_idxs = torch.tensor(batch_x, dtype=torch.long)

                    # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                    # new instance, you need to zero out the gradients from the old
                    # instance
                    model_network.zero_grad()

                    # Step 3. Run the forward pass, getting log probabilities over next
                    # words
                    predicted = model_network(context_idxs)

                    # Step 4. Compute your loss function. (Again, Torch wants the target
                    # word wrapped in a tensor)
                    loss = loss_function(predicted, torch.tensor(batch_y, dtype=torch.long))

                    # Step 5. Do the backward pass and update the gradient
                    loss.backward()
                    optimizer.step()

                    # Get the Python number from a 1-element Tensor by calling tensor.item()
                    total_loss += loss.item()
                    losses.append(total_loss)
        print(losses)  # The loss decreased
