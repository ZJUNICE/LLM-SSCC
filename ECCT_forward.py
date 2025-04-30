from __future__ import print_function
import argparse
import random
import os
import numpy as np
from torch.utils import data
import logging
import torch
import time
from pyldpc import make_ldpc
from utils import SNR_to_std, bin_to_sign, sign_to_bin, BER, FER, set_seed


##################################################################

class My_Dataset(data.Dataset):
    def __init__(self, message, code, sigma):
        self.message = message
        self.code = code
        self.sigma = sigma
        self.generator_matrix = code.generator_matrix.transpose(0, 1)  # Matrix G, shape(k, n)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)  # Matrix transpose(H), shape(n, l)

    def __getitem__(self, index):
        # Check message length, split into groups of length code.k
        num_groups, message_list = split_message(self.message, self.code.k)
        m_list, x_list, z_list, y_list, magnitude_list, syndrome_list = [], [], [], [], [], []
        for i in range(num_groups):
            single_msg = message_list[i]
            
            # transmit each group of message into a tensor of shape (1, code.k)
            m = torch.tensor(single_msg).view(1, self.code.k)
            x = torch.matmul(m, self.generator_matrix) % 2  # encoded codeword
            z = torch.randn(self.code.n) * self.sigma[0]  # noise
            
            if channel == 'AWGN':
                h = 1
            elif channel == 'Rayleigh':
                h = torch.from_numpy(np.random.rayleigh(1, self.code.n)).float()
            else:
                raise ValueError("Invalid channel type.")
            
            y = h * bin_to_sign(x) + z  # received signal
            
            
            magnitude = torch.abs(y)
            syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                    self.pc_matrix) % 2
            syndrome = bin_to_sign(syndrome)
            
            m_list.append(m.float())
            x_list.append(x.float())
            z_list.append(z.float())
            y_list.append(y.float())
            magnitude_list.append(magnitude.float())
            syndrome_list.append(syndrome.float())
            
        return m_list, x_list, z_list, y_list, magnitude_list, syndrome_list

    def __len__(self):
        return 1  # only 1 batch


##################################################################
##################################################################

def read_message_from_txt(filename):
    message_list = []
    with open(filename, 'r') as f:
        message_len, codeword_len = 0, 0
        message_string_list = f.read().splitlines()
        for message_string in message_string_list:
            message = [int(x) for x in message_string]
            message_list.append(message)
            message_len += len(message)
            codeword_len += (len(message)//code.k + 1) * code.n
    return message_list, message_len, codeword_len  # message

def split_message(message, k):
    """
    Splits the message list into sublists of length k. 
    If the last sublist is shorter than k, pads it with zeros. 
    Returns: 
        A list containing the sublists.
    """
    n = len(message)
    num_groups = (n + k - 1) // k
    result = []
    for i in range(num_groups):
        start = i * k
        end = min((i + 1) * k, n)
        group = message[start:end]
        if len(group) < k:
            group.extend([0] * (k - len(group)))
        result.append(group)
    return num_groups, result

##################################################################

def estimate(model, device, dataloader_list, SNR_range_test, code):
    model.eval()
    test_loss_noise_ber, test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], [], []
    t = time.time()
    with torch.no_grad():
        for ii in range(len(dataloader_list)):  # num of [std_test] list
            test_loader = dataloader_list[ii]
            noise_ber = test_loss = test_ber = test_fer = cum_count = 0.
            x_pred_list = []
            #########
            (m_list, x_list, z_list, y_list, magnitude_list, syndrome_list) = next(iter(test_loader))
            assert len(m_list) == len(x_list) == len(z_list) == len(y_list) == len(magnitude_list) == len(syndrome_list),\
                   "Length of lists in the test_dataloader inconsistent."
            for jj in range(len(m_list)):
                m, x, z, y, magnitude, syndrome = m_list[jj], x_list[jj], z_list[jj], y_list[jj], magnitude_list[jj], syndrome_list[jj]
                noise_ber += BER((y<=0).float(), x)
                z_mul = (y * bin_to_sign(x))
                
                z_pred = model(magnitude.to(device), syndrome.to(device))
                loss, x_pred = model.loss(z_pred, z_mul.to(device), y.to(device))

                test_loss += loss.item() * x.shape[0]

                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                x_pred_list.append(x_pred.squeeze(0).cpu().numpy().astype(int))
            #########
            
            cum_samples_all.append(cum_count)
            test_loss_noise_ber.append(noise_ber / cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            # print(f'Test SNR={SNR_range_test[ii]}, BER={test_loss_ber_list[-1]:.2e}')
            
            # Save ECCT predicted messages after channel transmission
            code_name = str(code.code_type)+'_K'+str(code.k)+'_N'+str(code.n)
            output_filename = os.path.join('mydata', 'ECCT_pred', code_name, channel, 'demo_decode_SNR_'+str(SNR_range_test[ii])+'.txt')
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            with open(output_filename, 'a') as f:
                combined_x_pred = np.concatenate(x_pred_list)
                f.write("".join(map(str, combined_x_pred))+"\n")
        logging.info('\nNoise BER ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr)
             in (zip(test_loss_noise_ber, SNR_range_test))]))
        logging.info('Test Loss ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr)
             in (zip(test_loss_list, SNR_range_test))]))
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr)
             in (zip(test_loss_fer_list, SNR_range_test))]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(snr, elem) for (elem, snr)
             in (zip(test_loss_ber_list, SNR_range_test))]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.2e}'.format(snr, -np.log(elem)) for (elem, snr)
             in (zip(test_loss_ber_list, SNR_range_test))]))
    logging.info(f'Test Time {time.time() - t} s\n')

    return test_loss_list, test_loss_ber_list, test_loss_fer_list

##################################################################
class Code():
    pass

def main(args):
    code = args.code
    message_list, message_len, codeword_len = read_message_from_txt(args.msg_path)
    logging.info(f'Message total length: {message_len}')
    logging.info(f'Codeword total length: {codeword_len}')
    
    message_unified_path = 'mydata/demo_encode_out_unified.txt'
    _,_,codeword_len_unified = read_message_from_txt(message_unified_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    if args.isParallel:
        model = torch.load(os.path.join(args.model_path, 'best_model'), map_location='cpu') # or map_location='cuda:0'
        model = model.module.to(device)
    else:
        model = torch.load(os.path.join(args.model_path, 'best_model'))
        model.to(device)
    
    logging.info(f'Transmission channel type: {args.channel}')
    # logging.info(model)
    #################################
    SNR_range_test = np.arange(-6, 16, 3)
    SNR_range_test_real = SNR_range_test + 10*np.log10(codeword_len_unified/codeword_len)
    # SNR_range_test_real = SNR_range_test
    print(f'SNR_range_test: {SNR_range_test_real}')
    
    std_test = [SNR_to_std(ii) for ii in SNR_range_test_real]
    print(f'std_test: {std_test}')
    
    test_loss_sum, test_loss_ber_sum, test_loss_fer_sum = [0] * len(std_test), [0] * len(std_test), [0] * len(std_test)
    for message in message_list:
        dataloader_list = [My_Dataset(message, code, sigma=[std_test[ii]]) for ii in range(len(std_test))]
        
        test_loss_list, test_loss_ber_list, test_loss_fer_list = estimate(model, device, dataloader_list, SNR_range_test, code)
        
        test_loss_sum = list(map(lambda x, y: x + y, test_loss_list, test_loss_sum))
        test_loss_ber_sum = list(map(lambda x, y: x + y, test_loss_ber_list, test_loss_ber_sum))
        test_loss_fer_sum = list(map(lambda x, y: x + y, test_loss_fer_list, test_loss_fer_sum))
    test_loss_avg = list(map(lambda x: x / len(message_list), test_loss_sum))
    test_loss_ber_avg = list(map(lambda x: x / len(message_list), test_loss_ber_sum))
    test_loss_fer_avg = list(map(lambda x: x / len(message_list), test_loss_fer_sum))
    
    code_name = str(code.code_type)+'_K'+str(code.k)+'_N'+str(code.n)
    metric_filename = os.path.join('mydata', 'ECCT_pred', code_name, channel, '#demo_decode_metric.txt')
    with open(metric_filename, "a") as f:
        f.write("\nloss:\n"+ ", ".join([f"{x:.8e}" for x in test_loss_avg]))
        f.write("\nBER:\n"+ ", ".join([f"{x:.8e}" for x in test_loss_ber_avg]))
        f.write("\nFER:\n"+ ", ".join([f"{x:.8e}" for x in test_loss_fer_avg]))
##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ECCT')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default="0", help='gpus ids')
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    # Message args
    parser.add_argument('--msg_filename', type=str, default='demo_encode_out')

    # Code args
    parser.add_argument('--code_type', type=str, default='LDPC',
                        choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=24)
    parser.add_argument('--code_n', type=int, default=49)
    parser.add_argument('--channel', type=str, default='Rayleigh')

    # Model args
    parser.add_argument('--isParallel', type=bool, default=True)  # DataParallel or not
    
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    # d_c: dimension of check nodes, d_v: dimension of variable nodes
    # d_c, d_v = 7, 4
    d_c = int(np.sqrt(code.n))
    d_v = int(d_c + 1 - code.k/(d_c-1))
    H, G = make_ldpc(code.n, d_v, d_c, systematic=True, sparse=True)
    code.generator_matrix = torch.from_numpy(G).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    channel = args.channel
    
    ####################################################################
    
    model_dir = os.path.join('Results_ECCT',
                             args.code_type + '__Code_n_' + str(
                                 args.code_n) + '_k_' + str(
                                 args.code_k) + '_' + args.channel)  ###
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model_dir {model_dir} does not exist.")
    else:
        args.model_path = model_dir
    
    msg_filename = args.msg_filename
    msg_dir = os.path.join('mydata', args.msg_filename + '.txt')
    if not os.path.exists(msg_dir):
        raise FileNotFoundError(f"Message_dir {msg_dir} does not exist.")
    else:
        args.msg_path = msg_dir
    
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging_test.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)
    
    main(args)