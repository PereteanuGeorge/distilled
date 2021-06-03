import time

import numpy as np
import tenseal as ts
import torch
from server import enc_model
from conv_fwd import conv, EncServer2
from dla_simple import DLA, PartConv
from utils import load_input, context, device, load_weights
import math

criterion = torch.nn.CrossEntropyLoss()

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])



if __name__ == "__main__":
    model = DLA()
    model.load_state_dict(torch.load("new_model_wih_kernel_1.pt"))
    weight = model.base.weight
    weight_R = weight[:, 0, :, :]
    weight_G = weight[:, 1, :, :]
    weight_B = weight[:, 2, :, :]
    print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! weight_R shape {weight_R.shape}')
    # aici era (16, 1, 1, 1)
    weight_R = weight_R.reshape(16, 1, 1, 1)
    weight_G = weight_G.reshape(16, 1, 1, 1)
    weight_B = weight_B.reshape(16, 1, 1, 1)

    img, target = load_input()
    img, target = img.to(device), target.to(device)
    R = img[:, 0, :, :]
    G = img[:, 1, :, :]
    B = img[:, 2, :, :]
    R = R.reshape(1, 1, 32, 32).to(device)
    G = G.reshape(1, 1, 32, 32).to(device)
    B = B.reshape(1, 1, 32, 32).to(device)

    first_conv = PartConv().to(device)
    load_weights(first_conv)

    start = time.time()
    with torch.no_grad():
        conv.base.weight.data = weight_R

    x_enc, windows_nb = ts.im2col_encoding(
        context, R.view(32, 32).tolist(), 1,
        1, 1
    )
    # aici era 1 16 32 32
    first_part = EncServer2(conv).to(device)
    output_R = first_part(x_enc, windows_nb)
    output_R = output_R.decrypt()
    output_R = np.reshape(output_R, (1, 16, 32, 32))
    output_R = torch.from_numpy(output_R).float()

    with torch.no_grad():
        conv.base.weight.data = weight_G

    x_enc, windows_nb = ts.im2col_encoding(
        context, G.view(32, 32).tolist(), 1,
        1, 1
    )
    first_part = EncServer2(conv).to(device)
    output_G = first_part(x_enc, windows_nb)
    output_G = output_G.decrypt()
    output_G = np.reshape(output_G, (1, 16, 32, 32))
    output_G = torch.from_numpy(output_G).float()

    with torch.no_grad():
        conv.base.weight.data = weight_B
    start_time = time.time() # asta nu era aici
    x_enc, windows_nb = ts.im2col_encoding(
        context, B.view(32, 32).tolist(), 1,
        1, 1
    )
    end_time = time.time()
    print(f'T1: time to encrypt the input image {end_time - start_time}')
    print(f'CT1: size of encrypted image {convert_size(len(x_enc.serialize()))}')
    first_part = EncServer2(conv).to(device)
    start_time1 = time.time()
    output_B = first_part(x_enc, windows_nb)
    end_time1 = time.time()
    print(f'T2: time to run the encrypted conv {end_time1 - start_time1}')
    print(f'CT2: size of encrypted weights after conv {convert_size(len(output_B.serialize()))}')
    output_B = output_B.decrypt()
    output_B = np.reshape(output_B, (1, 16, 32, 32))
    output_B = torch.from_numpy(output_B).float()
    #end_time = time.time()
    #print(f'!!!!!!!!!!!! o operatie a luat {end_time - start_time}')
    output_final = output_R + output_G + output_B
    output_final = output_final.to(device)

    #correct = 0
    #total = 0
    start = time.time()
    outputs = first_conv(output_final)
    #loss = criterion(outputs, target)
    x_enc = [ts.ckks_vector(context, x.tolist()) for x in outputs]
    end = time.time()
    print(f'T3: forward pass in plaintext and encrypt {end-start}')
    print(f'CT3 size {convert_size(len(x_enc))}')
    start = time.time()
    enc_output = enc_model(x_enc)
    result = enc_output.decrypt()
    #_, predicted = outputs.max(1)
    end = time.time()
    print(f'T4: {end-start}')

    #pytorch_total_params1 = sum(p.numel() for p in model1.parameters())
    #pytorch_total_params2 = sum(p.numel() for p in model2.parameters())
    #print(f'total number of params is {pytorch_total_params1 + pytorch_total_params2}')
