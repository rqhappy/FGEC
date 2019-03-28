import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # arguments for project
    parser.add_argument('-seed', type=int,
                        help='manual seed for initializations', default=5)
    parser.add_argument('-exp', '--experiment_root', type=str, default='./output',
                        help='root where to store models, losses and accuracies')

    # arguments for training settings
    parser.add_argument('-cuda', action='store_true', help='enable cuda')
    parser.add_argument('-epochs', type=int, default=40)
    parser.add_argument('-lr', '--learning_rate', type=float, default=.001)
    parser.add_argument('-batch_size', type=int, default=1000)
    parser.add_argument('-corpus_dir', type=str, choices=["Wiki/", "OntoNotes/"], default='Wiki/')

    # arguments for learning rate scheduler
    parser.add_argument('-lrs', '--learning_scheduler_step', type=int,
                        help='StepLR learning rate scheduler step', default=10)
    parser.add_argument('-lrg', '--learning_scheduler_gamma', type=float,
                        help='StepLR learning rate scheduler gamma', default=0.5)

    return parser
