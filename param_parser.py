import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run CapsGNN.")
    str2bool = lambda x: x.lower() == "true"

    parser.add_argument('--exp', type=str, default="test")
    parser.add_argument('--data_root', type=str, default="data")
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epoch_select', type=str, default='test_max', 
                        help="{test_max, val_min} test_max: select a single epoch; \
                        val_min: select epoch with the lowest val loss.")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--with_eval_mode', type=str2bool, default=True)
    
    parser.add_argument("--no-cuda",
                        default=False,
                        action="store_true",
                    help="Not use CUDA.")

    parser.add_argument('--dataset', 
                        type=str, 
                        default="MUTAG",
                    help="The name of the dataset.")

    parser.add_argument("--disentangle-num",
                        type=int,
                        default=4,
                    help="Disentangle feature num. Default is 4.")

    parser.add_argument("--capsule-num",
                        type=int,
                        default=10,
                    help="Graph capsule num. Default is 10.")

    parser.add_argument("--capsule-dimensions",
                        type=int,
                        default=128,
                    help="Capsule dimensions. Default is 128.")

    parser.add_argument("--num-iterations",
                        type=int,
                        default=3,
                    help="Number of routing iterations. Default is 3.")

    parser.add_argument("--theta",
                        type=float,
                        default=0.1,
                    help="Reconstruction loss weight. Default is 0.1.")

    parser.add_argument("--seed",
                        type=int,
                        default=72,
                    help="Random seed. Default is 72.")

    parser.add_argument('--log-path', 
                        type=str, 
                        default="log/train_loss",
                    help="The path of training log.")

    return parser.parse_args()
