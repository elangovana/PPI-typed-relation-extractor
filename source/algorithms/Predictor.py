import torch


class Predictor:

    def predict(self, model_network, dataloader):
        # switch model to evaluation mode
        model_network.eval()

        predicted = []
        scores = []
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                val_batch_idx = batch[0]
                pred_batch_y = model_network(val_batch_idx)
                scores.append(pred_batch_y)
                pred_binary = torch.max(pred_batch_y, 1)[1]
                pred_flat = pred_binary.view(pred_binary.size())

                predicted.append(pred_flat.numpy().tolist())

        scores = [r.numpy().tolist() for r in scores]
        return predicted, scores
