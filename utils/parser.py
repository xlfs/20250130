import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="mimic3")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )
    parser.add_argument("--model", default="boxlm")
    parser.add_argument("--pretrain_model_path", default="")
    parser.add_argument("--graph_type", default="all", help="choose in [all, visit, ccs, none]")
    parser.add_argument("--eval_rnd_neg", action="store_true", default=False)
    parser.add_argument("--tau", default=1.0, type=float)
    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=2000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=16, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--inverse_r", action="store_false", default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 30]', help='Output sizes of every layer')
    parser.add_argument("--eval_step", default=5, type=int)
    parser.add_argument("--beta", default=0.2, type=float)
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--save",  default=False, action="store_true", help="save the model")
    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=1, help='number of context hops')
    # ===== save model ===== #
    parser.add_argument("--logit_cal", type=str, default="box")
    return parser.parse_args()
