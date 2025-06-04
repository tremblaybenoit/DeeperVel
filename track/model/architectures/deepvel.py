import torchvision
from torch import nn
from track.utils.instantiators import instantiate


class DeepVelModel(nn.Module):

    def __init__(self, d_input, d_output, version, dp=0.75):
        """ Initialize EfficientNet neural network model. Define functions to overwrite BaseModel.

            Parameters
            ----------
            d_input: int. Dimensions of input vector.
            d_output: int. Dimensions of output vector.
            version: str. Choice of EfficientNet model. Use pretrained EfficientNet or train from scratch.
            dp: float. Dropout fraction.

            Returns
            -------
            EfficientNet architecture.
        """

        # Class inheritance
        super().__init__()

        # Choice of EfficientNet model
        model = instantiate(version)
        # Account for input/output dimensions
        conv1_out = model.features[0][0].out_channels
        model.features[0][0] = nn.Conv2d(d_input, conv1_out, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                         bias=False)
        lin_in = model.classifier[1].in_features
        # consider adding average pool of full image(s)
        classifier = nn.Sequential(nn.Dropout(p=dp, inplace=True),
                                   nn.Linear(in_features=lin_in, out_features=d_output, bias=True))
        model.classifier = classifier
        # Dropout
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p = dp
        # Store
        self.model = model

    def forward(self, x):
        """ Pass forward through neural network architecture.

            Parameters
            ----------
            x: tensor. Inputs.

            Returns
            -------
            y: tensor. Outputs.
        """
        x = self.model(x)
        return x
