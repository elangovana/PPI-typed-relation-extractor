import torch


class Predictor:

    def predict(self, model_network, dataloader, device=None):
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_network.to(device)
        # switch model to evaluation mode
        model_network.eval()
        predicted = []
        scores = []
        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(dataloader):

                # TODO: CLean this up
                if isinstance(batch_x, list):
                    val_batch_idx = [t.to(device=device) for t in batch_x]
                else:
                    val_batch_idx = batch_x.to(device=device)

                pred_batch_y = model_network(val_batch_idx)
                scores.append(pred_batch_y)
                pred_binary = torch.max(pred_batch_y, 1)[1]
                pred_flat = pred_binary.view(pred_binary.size())

                predicted.append(pred_flat.cpu().numpy().tolist())

        scores = [r.cpu().numpy().tolist() for r in scores]
        return predicted, scores
