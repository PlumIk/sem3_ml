from cnn import train

if __name__ == '__main__':
    parameters_file_name = "params/data.pkl"

    train.train(
        lr=0.01,
        gamma=0.95,
        batch_size=50,
        parameters_file_name=parameters_file_name)
