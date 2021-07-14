import pandas as pd
import tensorflow as tf
import Network as nt
import DiscordBot as db


class GenerateText(db.Command):
    def __init__(self, folder, argn=2):
        db.Command.__init__(self, argn)
        indx = pd.read_csv(folder + "/index")  # Read model data
        model = nt.Network(indx["inp_size"], indx["emb_size"], indx["rnn_size"], 0.0)  # Create model object
        df, _, _ = nt.prepare_batches(indx["init_seq"], 1, indx["seq_len"])  # Generate data for model init
        model.compile_and_train(df, epochs=1, checkpoints=False,
                                end_model_name=None)  # Run short training for model initializing
        model.load(folder + "/weight")  # Read model weight
        _, ifc, cfi = nt.preprocess(indx["chrs"])  # Extracting list of chars used in model
        self.text_gen = nt.TextGenerator(model, cfi, ifc)

    def __call__(self, argv, message, client):
        db.Command.__call__(argv, message, client)
        states = None
        next_char = tf.constant([" ".join(argv[1:])])  # Create initial sequence
        result = [next_char]

        for n in range(int(argv[0])):
            # Iterate using previous iteration output as input
            next_char, states = self.text_gen.generate_one_step(next_char, states=states)
            result.append(next_char)
        # Join first symbol from each output sequence
        await message.channel.send(tf.strings.join(result)[0].numpy().decode("utf-8"))
        return None
