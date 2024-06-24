if __name__ == '__main__':

    import argparse
    import os
    import torch
    from exp.exp_main import Exp_Main
    from utils.tools import log_string
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='TRECK with seq2seq RNN and BiLSTM')

    parser.add_argument('--model', type=str, required=False, default='rnn_bilstm',
                        help='model name, options: [rnn, rnn_bilstm]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='TRECK', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='1115432withHoli_hour_holifix.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='volume', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')   
    parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length') 

    parser.add_argument('--embed_type', type=int, default=0, 
                        help='0: contrasive embedding, 1: crafted time features')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')     
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of hidden cell') 
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')  
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    # contrasive learning parameters
    parser.add_argument('--beta', type=float, default=0.000001, help='Weighting hyperparameter for loss function')
    parser.add_argument('--sigma_q', type=float, default=0, help='adjust the jittering for positive sample')
    parser.add_argument('--sigma_k', type=float, default=5, help='adjust the jittering for positive sample')
    parser.add_argument('--band_num', type=float, default=1, help='number of bands')

    args = parser.parse_args(args=[])

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    Exp = Exp_Main

    for ii in range(args.itr):
        setting = '{}_embed{}_bat{}_{}_lr{}_enc{}_sl{}_pl{}_dm{}_el{}_dl{}_al{}_sigq{}_sigk{}_bn{}_{}'.format(
            args.model,
            args.embed_type,
            args.batch_size,
            args.data_path[:7],
            args.learning_rate,
            args.enc_in,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.beta,
            args.sigma_q,
            args.sigma_k,
            args.band_num,
        ii)        
        now = datetime.now()
        exp_path = './results/' + setting
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        
        log = open (exp_path + '/log.txt', 'w')
        log_string(log, 'Args in experiment:')
        log_string(log, str (args))

        exp = Exp(args)
        log_string(log,'>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting,log)

        log_string(log,'>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting,log,exp_path)
        
        log.close()

        torch.cuda.empty_cache()
