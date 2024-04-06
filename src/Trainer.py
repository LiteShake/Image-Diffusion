
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

class Trainer :

    def __init__(self, data = None, model = None) -> None:

        self.data = data
        self.model = model

    def create_schedule(self, image, idx):

        torch.manual_seed(0)
        tape = []
        image = np.float32(image)
        image /= 255.0

        SPACE = np.linspace(0, np.pi / 2, 100)
        COS_SPACE = np.cos(SPACE)

        noised = image

        fig, ax = plt.subplots(1, 100)
        fig.set_figheight(5)
        fig.set_figwidth(600)


        for i in range(100):

            tape.append(torch.tensor(noised))
            print(noised.shape)
            ax[i].imshow( (torch.tensor(noised)[0]).permute(1, 2, 0) )

            noise = np.random.normal(COS_SPACE[i], 12 , noised.shape)
            noise /= 127

            noised += noise

            noised = np.clip(noised, 0, 1)


        plt.savefig(f"hotlady{idx}.png")
        #plt.show()


        return tape[::-1]

    def train(self, device):

        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2, weight_decay = 0.03)
        self.model = self.model.to(device)

        logger = SummaryWriter("./runs/UNet Test 1")

        for batch_num in range(0, len(self.data), 10):

            tapes = []
            batch = self.data[batch_num : batch_num + 10]

            idx = 0

            for img in batch:

                tapes.append(self.create_schedule(img, idx))
                idx += 1

            print(len(tapes))

            for idx in range(len(tapes[0]) - 1):

                tape_idx = 0
                # loop over the dataset multiple times
                for tape in tapes:

                    tape_idx += 1
                    running_loss = 0.0

                    for epoch in range(100):


                        inputs, labels = tape[idx], tape[idx + 1]

                        # print(f"Inputs {type(inputs)} Labels {type(labels)}")

                        inputs, labels = inputs.float(), (inputs - labels).float()
                        inputs, labels = inputs.to(device), labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize


                        outputs = self.model(inputs, idx + 1)

                        # fig, ax = plt.subplots(1, 3)
                        # ax[0].imshow( (inputs[0].cpu().detach()).permute(1, 2, 0) )
                        # ax[1].imshow( (labels[0].cpu().detach()).permute(1, 2, 0) / 225 )
                        # ax[2].imshow( (outputs[0].cpu().detach()).permute(1, 2, 0) )

                        # plt.show()

                        # print(outputs)

                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # running_loss += loss.item()

                        print(f"Batch {batch_num} Tape {tape_idx} Tape Loc {idx} Epoch {epoch} Loss {loss.item()}")

                    logger.add_scalar("Running Loss", running_loss, batch_num)


        torch.save(self.model.state_dict(), "./Models/Saves/UNet1.pt")
