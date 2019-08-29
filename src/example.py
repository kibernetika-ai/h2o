import argparse

import h2o


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('host')
    parser.add_argument(
        '--port',
        type=int,
        default='80'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    h2o.init(ip=args.host, port=args.port)

    # Upload the prostate dataset that comes included in the h2o python package
    prostate = h2o.load_dataset("prostate")

    # Print a description of the prostate data
    prostate.describe()

    # Randomly split the dataset into ~70/30, training/test sets
    train, test = prostate.split_frame(ratios=[0.70])

    # Convert the response columns to factors (for binary classification problems)
    train["CAPSULE"] = train["CAPSULE"].asfactor()
    test["CAPSULE"] = test["CAPSULE"].asfactor()

    # Build a (classification) GLM
    from h2o.estimators import H2OGradientBoostingEstimator
    prostate_gbm = H2OGradientBoostingEstimator(distribution="bernoulli", ntrees=10, max_depth=8,
                                                min_rows=10, learn_rate=0.2)
    prostate_gbm.train(x=["AGE", "RACE", "PSA", "VOL", "GLEASON"],
                       y="CAPSULE", training_frame=train)

    # Show the model
    prostate_gbm.show()

    # Predict on the test set and show the first ten predictions
    predictions = prostate_gbm.predict(test)
    predictions.show()

    # Fetch a tree, print number of tree nodes, show root node description
    from h2o.tree import H2OTree, H2ONode
    tree = H2OTree(prostate_gbm, 0, "0")
    tree.root_node.show()

    # Show default performance metrics
    performance = prostate_gbm.model_performance(test)
    performance.show()


if __name__ == '__main__':
    main()
