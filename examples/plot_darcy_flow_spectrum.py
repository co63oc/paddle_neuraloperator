import sys
sys.path.append('/nfs/github/paddle/paddle_neuraloperator/utils')
import paddle_aux
import paddle
"""
A simple Darcy-Flow spectrum analysis
=====================================
In this example, we demonstrate how to use the spectrum analysis function on the small Darcy-Flow example.
For more details on spectrum analysis, users can take a look at this reference: https://www.astronomy.ohio-state.edu/ryden.1/ast825/ch7.pdf

Short summary
-------------

Spectral analysis is useful because it allows researchers to study the distribution of energy across different scales in a fluid flow. By examining the energy spectrum, one can gain insights into the behavior of turbulence or any other dataset and the underlying physical processes. The energy spectrum is analysed through the Fourier transform which is a mathematical tool that decomposes a function or signal into its constituent frequencies. In a fluid flow, it is used to analyze the distribution of energy across different scales in a flow. Specifically, the Fourier transform is applied to the velocity field of the flow, converting it into a frequency domain representation. Higher the wavenumber corresponds to higher frequency and higher energy and is a much harder task to solve as we need higher modes to capture the high-frequency behavior of the flow. Overall this allows researchers to study the energy spectrum, which provides insights into the behavior of turbulence and the underlying physical processes.

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neuralop.utils import spectrum_2d
from neuralop.data.datasets import load_darcy_flow_small
font = {'size': 28}
matplotlib.rc('font', **font)
paddle.seed(seed=0)
np.random.seed(0)
T = 500
samples = 50
s = 16
Re = 5000
index = 1
T = 100
dataset_name = 'Darcy Flow'
train_loader, test_loaders, data_processor = load_darcy_flow_small(n_train=
    50, batch_size=50, test_resolutions=[16, 32], n_tests=[50],
    test_batch_sizes=[32], encode_output=False)
print('Original dataset shape', tuple(train_loader.dataset[:samples]['x'].
    shape))
dataset_pred = train_loader.dataset[:samples]['x'].squeeze()
shape = tuple(dataset_pred.shape)
"""
batchsize, size_x, size_y, size_z = 1, shape[0], shape[1], shape[2]
gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float)
gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
gridz = torch.tensor(np.linspace(-1, 1, size_z), dtype=torch.float)
gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
grid = torch.cat((gridx, gridy, gridz), dim=-1)
"""
batchsize, size_x, size_y = 1, shape[1], shape[2]
gridx = paddle.to_tensor(data=np.linspace(-1, 1, size_x), dtype='float32')
gridx = gridx.reshape(1, size_x, 1).tile(repeat_times=[batchsize, 1, size_y])
gridy = paddle.to_tensor(data=np.linspace(-1, 1, size_y), dtype='float32')
gridy = gridy.reshape(1, 1, size_y).tile(repeat_times=[batchsize, size_x, 1])
grid = paddle.concat(x=(gridx, gridy), axis=-1)
truth_sp = spectrum_2d(dataset_pred.reshape(samples * batchsize, s, s), s)
fig, ax = plt.subplots(figsize=(10, 10))
linewidth = 3
ax.set_yscale('log')
length = 16
buffer = 10
k = np.arange(length + buffer) * 1.0
ax.plot(truth_sp, 'k', linestyle=':', label='NS', linewidth=4)
ax.set_xlim(1, length + buffer)
ax.set_ylim(10, 10 ^ 10)
plt.legend(prop={'size': 20})
plt.title('Spectrum of {} Datset'.format(dataset_name))
plt.xlabel('wavenumber')
plt.ylabel('energy')
leg = plt.legend(loc='best')
leg.get_frame().set_alpha(0.5)
plt.show()
